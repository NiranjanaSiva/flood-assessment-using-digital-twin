import React, { useEffect, useRef, useState } from "react";

import * as maplibregl from "maplibre-gl";
import { setWorkerUrl } from "maplibre-gl";

import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import "maplibre-gl/dist/maplibre-gl.css";

// ---------------------------------------------------------
// MAPLIBRE WORKER
// ---------------------------------------------------------

setWorkerUrl(workerUrl);

// ---------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------

const BACKEND_URL = "http://127.0.0.1:8000";

const CSV_FILE = "/AIFloodSense-test.csv";

const MAP_STYLE = "https://tiles.openfreemap.org/styles/liberty";

const TERRAIN_URL = "https://tiles.mapterhorn.com/tilejson.json";

// WorldPop population estimate settings.
// WorldPop v2 supports global population statistics at 100 m resolution.
const WORLDPOP_API = "https://api.worldpop.org/v2";
const WORLDPOP_YEAR = 2026;
const WORLDPOP_RESOLUTION = "100m";

// Keep the map readable if the downhill simulation creates many tiny
// disconnected predicted areas. The largest components get markers first.
const MAX_POPULATION_ZONES = 20;
const MIN_POPULATION_ZONE_CELLS = 2;

// Approximate geographic size of the uploaded aerial image.
// This is used because the image itself does not contain
// geographic bounds.
const IMAGE_WIDTH_DEGREES = 0.02;
const IMAGE_HEIGHT_DEGREES = 0.015;

// ---------------------------------------------------------
// CSV PARSER
// ---------------------------------------------------------

function parseCSV(text) {
  const rows = [];

  let row = [];
  let value = "";

  let insideQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    // Handle quotes
    if (char === '"') {
      if (insideQuotes && next === '"') {
        value += '"';
        i++;
      } else {
        insideQuotes = !insideQuotes;
      }

      continue;
    }

    // Comma separates values
    if (char === "," && !insideQuotes) {
      row.push(value.trim());
      value = "";

      continue;
    }

    // New line separates rows
    if ((char === "\n" || char === "\r") && !insideQuotes) {
      if (char === "\r" && next === "\n") {
        i++;
      }

      row.push(value.trim());

      if (row.some((v) => v !== "")) {
        rows.push(row);
      }

      row = [];
      value = "";

      continue;
    }

    value += char;
  }

  // Last value
  if (value !== "" || row.length > 0) {
    row.push(value.trim());

    if (row.some((v) => v !== "")) {
      rows.push(row);
    }
  }

  if (rows.length === 0) {
    return [];
  }

  const headers = rows[0].map((header) => header.trim());

  const result = [];

  for (let i = 1; i < rows.length; i++) {
    const object = {};

    for (let j = 0; j < headers.length; j++) {
      object[headers[j]] = rows[i][j] !== undefined ? rows[i][j] : "";
    }

    result.push(object);
  }

  return result;
}

// ---------------------------------------------------------
// HEADER NORMALIZATION
// ---------------------------------------------------------

function normalizeHeader(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/[\s_-]+/g, "");
}

// ---------------------------------------------------------
// GET COLUMN
// ---------------------------------------------------------

function findColumn(row, possibleNames) {
  const keys = Object.keys(row);

  for (const name of possibleNames) {
    const normalizedName = normalizeHeader(name);

    const key = keys.find((k) => normalizeHeader(k) === normalizedName);

    if (key) {
      return key;
    }
  }

  return null;
}

// ---------------------------------------------------------
// IMAGE ID
// ---------------------------------------------------------

function getImageId(filename) {
  const name = filename.split("\\").pop().split("/").pop();

  return name
    .replace(/\.[^.]+$/, "")
    .trim()
    .replace(/^0+/, "");
}

// ---------------------------------------------------------
// COORDINATE PARSER
// ---------------------------------------------------------

function parseCoordinate(value, type) {
  if (value === undefined || value === null) {
    return NaN;
  }

  let text = String(value).trim().toUpperCase();

  if (text === "") {
    return NaN;
  }

  // -------------------------------------------------------
  // IMPORTANT:
  //
  // CSV contains:
  //
  // 49,87
  // 18,28
  //
  // These mean:
  //
  // 49.87
  // 18.28
  //
  // Convert decimal comma to decimal point.
  // -------------------------------------------------------

  text = text.replace(",", ".");

  const hasSouth = text.includes("S");

  const hasWest = text.includes("W");

  const match = text.match(/[-+]?\d+(?:\.\d+)?/);

  if (!match) {
    return NaN;
  }

  let number = Number(match[0]);

  if (!Number.isFinite(number)) {
    return NaN;
  }

  // Direction
  if (type === "latitude" && hasSouth) {
    number = -Math.abs(number);
  }

  if (type === "longitude" && hasWest) {
    number = -Math.abs(number);
  }

  // Validate latitude
  if (type === "latitude" && (number < -90 || number > 90)) {
    return NaN;
  }

  // Validate longitude
  if (type === "longitude" && (number < -180 || number > 180)) {
    return NaN;
  }

  return number;
}

// ---------------------------------------------------------
// FIND IMAGE LOCATION
// ---------------------------------------------------------

function findImageLocation(rows, filename) {
  if (!rows || rows.length === 0) {
    console.error("CSV contains no rows.");

    return null;
  }

  const imageId = getImageId(filename);

  const firstRow = rows[0];

  const imageColumn = findColumn(firstRow, [
    "IMAGE_ID",
    "IMAGEID",
    "IMAGE",
    "ID",
  ]);

  const latitudeColumn = findColumn(firstRow, ["LATITUDE", "LAT", "Y"]);

  const longitudeColumn = findColumn(firstRow, [
    "LONGITUDE",
    "LON",
    "LONG",
    "LNG",
    "X",
  ]);

  console.log("Image column:", imageColumn);

  console.log("Latitude column:", latitudeColumn);

  console.log("Longitude column:", longitudeColumn);

  if (!imageColumn || !latitudeColumn || !longitudeColumn) {
    console.error("Required CSV columns not found.");

    return null;
  }

  console.log("Searching CSV for:", imageId);

  for (const row of rows) {
    const rawImageId = String(row[imageColumn] || "")
      .trim()
      .replace(/\.[^.]+$/, "")
      .replace(/^0+/, "");

    if (rawImageId !== imageId) {
      continue;
    }

    console.log("RAW CSV RECORD:", row);

    const rawLatitude = row[latitudeColumn];

    const rawLongitude = row[longitudeColumn];

    console.log("Raw latitude:", rawLatitude);

    console.log("Raw longitude:", rawLongitude);

    const latitude = parseCoordinate(rawLatitude, "latitude");

    const longitude = parseCoordinate(rawLongitude, "longitude");

    console.log("Parsed latitude:", latitude);

    console.log("Parsed longitude:", longitude);

    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      console.error("Invalid coordinates.");

      return null;
    }

    return {
      latitude,
      longitude,
      row,
    };
  }

  console.error("Image not found in CSV:", imageId);

  return null;
}

// ---------------------------------------------------------
// IMAGE BOUNDS
// ---------------------------------------------------------

function getImageBounds(latitude, longitude) {
  const west = longitude - IMAGE_WIDTH_DEGREES / 2;

  const east = longitude + IMAGE_WIDTH_DEGREES / 2;

  const south = latitude - IMAGE_HEIGHT_DEGREES / 2;

  const north = latitude + IMAGE_HEIGHT_DEGREES / 2;

  return [
    [west, south],
    [east, north],
  ];
}

// ---------------------------------------------------------
// REMOVE OLD FLOOD
// ---------------------------------------------------------

function removeFloodLayer(map) {
  try {
    if (map.getLayer("flood-water")) {
      map.removeLayer("flood-water");
    }
  } catch (error) {
    console.warn("Could not remove flood layer:", error);
  }

  try {
    if (map.getSource("flood-water")) {
      map.removeSource("flood-water");
    }
  } catch (error) {
    console.warn("Could not remove flood source:", error);
  }
}

// ---------------------------------------------------------
// LOAD IMAGE
// ---------------------------------------------------------

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();

    image.crossOrigin = "anonymous";

    image.onload = () => resolve(image);

    image.onerror = () =>
      reject(new Error("Could not load prediction image: " + url));

    image.src = url;
  });
}

// ---------------------------------------------------------
// FLOOD MASK → GEOJSON
// ---------------------------------------------------------

async function predictionToGeoJSON(imageUrl, latitude, longitude) {
  console.log("Loading prediction image:", imageUrl);

  const image = await loadImage(imageUrl);

  console.log("Prediction image size:", image.width, "x", image.height);

  // Keep processing manageable
  const MAX_SIZE = 128;

  const scale = Math.min(1, MAX_SIZE / Math.max(image.width, image.height));

  const width = Math.max(1, Math.floor(image.width * scale));

  const height = Math.max(1, Math.floor(image.height * scale));

  const canvas = document.createElement("canvas");

  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  context.drawImage(image, 0, 0, width, height);

  const imageData = context.getImageData(0, 0, width, height);

  const pixels = imageData.data;

  const bounds = getImageBounds(latitude, longitude);

  const west = bounds[0][0];

  const south = bounds[0][1];

  const east = bounds[1][0];

  const north = bounds[1][1];

  const cellWidth = (east - west) / width;

  const cellHeight = (north - south) / height;

  const features = [];

  const floodGrid = Array.from({ length: height }, () =>
    Array(width).fill(false),
  );

  let floodedPixels = 0;

  // -------------------------------------------------------
  // Read flood pixels
  // -------------------------------------------------------

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;

      const r = pixels[index];

      const g = pixels[index + 1];

      const b = pixels[index + 2];

      const a = pixels[index + 3];

      if (a < 10) {
        continue;
      }

      const brightness = (r + g + b) / 3;

      // White / bright pixels are flood
      if (brightness < 80) {
        continue;
      }

      floodedPixels++;

      floodGrid[y][x] = true;

      const x1 = west + x * cellWidth;

      const x2 = x1 + cellWidth;

      // Canvas Y is from top to bottom
      const y2 = north - y * cellHeight;

      const y1 = y2 - cellHeight;

      features.push({
        type: "Feature",
        properties: {
          flood: true,
        },
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [x1, y1],
              [x2, y1],
              [x2, y2],
              [x1, y2],
              [x1, y1],
            ],
          ],
        },
      });
    }
  }

  console.log("Flood pixels detected:", floodedPixels);

  console.log(
    "Flood percentage:",
    ((floodedPixels / (width * height)) * 100).toFixed(2) + "%",
  );

  if (features.length === 0) {
    console.warn("NO FLOOD PIXELS DETECTED.");

    return null;
  }

  return {
    geoJSON: {
      type: "FeatureCollection",
      features,
    },
    floodGrid,
    width,
    height,
    bounds: { west, south, east, north },
  };
}

// ---------------------------------------------------------
// TERRAIN FLOOD SIMULATION SETTINGS
// ---------------------------------------------------------

const TERRAIN_EXAGGERATION = 1.5;
const MAX_FLOW_DISTANCE_METERS = 300;
const MIN_ELEVATION_DROP_METERS = 0.25;
const MAX_FLOW_STEPS = 10;

// ---------------------------------------------------------
// WAIT FOR TERRAIN DEM
// ---------------------------------------------------------

async function waitForTerrain(map, longitude, latitude) {
  console.log("Waiting for terrain DEM...");

  const start = Date.now();
  const timeout = 10000;

  while (Date.now() - start < timeout) {
    try {
      const elevation = map.queryTerrainElevation([longitude, latitude]);

      if (
        elevation !== null &&
        elevation !== undefined &&
        Number.isFinite(elevation)
      ) {
        console.log("Terrain DEM is available.");
        return true;
      }
    } catch (error) {
      console.warn("Terrain query waiting:", error);
    }

    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  console.warn("Terrain DEM did not become available.");
  return false;
}

// ---------------------------------------------------------
// GET TERRAIN ELEVATION
// ---------------------------------------------------------

function getTerrainElevation(map, longitude, latitude) {
  try {
    const raw = map.queryTerrainElevation([longitude, latitude]);

    if (raw === null || raw === undefined || !Number.isFinite(raw)) {
      return null;
    }

    // MapLibre's terrain query reflects terrain exaggeration.
    return raw / TERRAIN_EXAGGERATION;
  } catch (error) {
    console.warn("Terrain elevation error:", error);
    return null;
  }
}

// ---------------------------------------------------------
// GPS DISTANCE
// ---------------------------------------------------------

function distanceMeters(lng1, lat1, lng2, lat2) {
  const earthRadius = 6371000;

  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;

  return earthRadius * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ---------------------------------------------------------
// CELL CENTER
// ---------------------------------------------------------

function getCellCenter(x, y, width, height, bounds) {
  const cellWidth = (bounds.east - bounds.west) / width;
  const cellHeight = (bounds.north - bounds.south) / height;

  return {
    longitude: bounds.west + (x + 0.5) * cellWidth,
    latitude: bounds.north - (y + 0.5) * cellHeight,
  };
}

// ---------------------------------------------------------
// GRID CELL → GEOJSON
// ---------------------------------------------------------

function gridCellToFeature(x, y, width, height, bounds, properties = {}) {
  const cellWidth = (bounds.east - bounds.west) / width;
  const cellHeight = (bounds.north - bounds.south) / height;

  const x1 = bounds.west + x * cellWidth;
  const x2 = x1 + cellWidth;
  const y2 = bounds.north - y * cellHeight;
  const y1 = y2 - cellHeight;

  return {
    type: "Feature",
    properties,
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [x1, y1],
          [x2, y1],
          [x2, y2],
          [x1, y2],
          [x1, y1],
        ],
      ],
    },
  };
}

// ---------------------------------------------------------
// BOUNDARY CELL
// ---------------------------------------------------------

function isBoundaryCell(grid, x, y) {
  const height = grid.length;
  const width = grid[0].length;

  if (!grid[y][x]) return false;

  const neighbours = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];

  for (const [dx, dy] of neighbours) {
    const nx = x + dx;
    const ny = y + dy;

    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
      return true;
    }

    if (!grid[ny][nx]) return true;
  }

  return false;
}

// ---------------------------------------------------------
// TERRAIN-BASED FLOOD FLOW
// ---------------------------------------------------------

async function simulateFloodFlow(map, floodData) {
  console.log("====================================");
  console.log("STARTING TERRAIN FLOOD FLOW");
  console.log("====================================");

  const { floodGrid, width, height, bounds } = floodData;

  const occupied = floodGrid.map((row) => [...row]);

  let frontier = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (isBoundaryCell(floodGrid, x, y)) {
        frontier.push({ x, y });
      }
    }
  }

  console.log("Initial flood boundary cells:", frontier.length);

  if (!frontier.length) {
    return { type: "FeatureCollection", features: [] };
  }

  const elevationCache = new Map();
  const flowDistance = new Map();
  const predictedCells = [];

  const key = (x, y) => `${x}_${y}`;

  const getElevation = (x, y) => {
    const k = key(x, y);

    if (elevationCache.has(k)) return elevationCache.get(k);

    const center = getCellCenter(x, y, width, height, bounds);
    const elevation = getTerrainElevation(
      map,
      center.longitude,
      center.latitude,
    );

    elevationCache.set(k, elevation);
    return elevation;
  };

  for (const cell of frontier) {
    flowDistance.set(key(cell.x, cell.y), 0);
  }

  const neighbours = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];

  for (let step = 0; step < MAX_FLOW_STEPS; step++) {
    console.log(`Flood flow step ${step + 1}/${MAX_FLOW_STEPS}`);

    const nextFrontier = [];
    let newCells = 0;

    for (const source of frontier) {
      const sourceElevation = getElevation(source.x, source.y);
      if (sourceElevation === null) continue;

      const sourceCenter = getCellCenter(
        source.x,
        source.y,
        width,
        height,
        bounds,
      );

      const sourceDistance = flowDistance.get(key(source.x, source.y)) || 0;

      for (const [dx, dy] of neighbours) {
        const nx = source.x + dx;
        const ny = source.y + dy;

        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        if (occupied[ny][nx]) continue;

        const candidateCenter = getCellCenter(nx, ny, width, height, bounds);

        const stepDistance = distanceMeters(
          sourceCenter.longitude,
          sourceCenter.latitude,
          candidateCenter.longitude,
          candidateCenter.latitude,
        );

        const totalDistance = sourceDistance + stepDistance;

        if (totalDistance > MAX_FLOW_DISTANCE_METERS) continue;

        const candidateElevation = getElevation(nx, ny);
        if (candidateElevation === null) continue;

        const elevationDrop = sourceElevation - candidateElevation;

        // The new cell must be lower than the cell the flood
        // is currently flowing from.
        if (elevationDrop < MIN_ELEVATION_DROP_METERS) continue;

        occupied[ny][nx] = true;
        flowDistance.set(key(nx, ny), totalDistance);

        nextFrontier.push({ x: nx, y: ny });

        predictedCells.push({
          x: nx,
          y: ny,
          elevation: candidateElevation,
          elevationDrop,
          distance: totalDistance,
        });

        newCells++;
      }
    }

    console.log("New downhill cells:", newCells);

    if (!nextFrontier.length) {
      console.log("No more lower terrain found.");
      break;
    }

    frontier = nextFrontier;
  }

  console.log("Total predicted cells:", predictedCells.length);

  return {
    geoJSON: {
      type: "FeatureCollection",
      features: predictedCells.map((cell) =>
        gridCellToFeature(cell.x, cell.y, width, height, bounds, {
          flood: true,
          source: "terrain-flow",
          elevation: cell.elevation,
          elevationDrop: cell.elevationDrop,
          distance: cell.distance,
        }),
      ),
    },
    predictedCells,
    width,
    height,
    bounds,
  };
}

// ---------------------------------------------------------
// ADD PREDICTED FLOOD LAYER
// ---------------------------------------------------------

function removePredictedFloodLayer(map) {
  try {
    if (map.getLayer("predicted-flood-water"))
      map.removeLayer("predicted-flood-water");
  } catch (error) {
    console.warn("Could not remove predicted flood layer:", error);
  }
  try {
    if (map.getSource("predicted-flood-water"))
      map.removeSource("predicted-flood-water");
  } catch (error) {
    console.warn("Could not remove predicted flood source:", error);
  }
}

function addPredictedFloodLayer(map, predictedGeoJSON) {
  try {
    if (map.getLayer("predicted-flood-water")) {
      map.removeLayer("predicted-flood-water");
    }
  } catch (error) {
    console.warn(error);
  }

  try {
    if (map.getSource("predicted-flood-water")) {
      map.removeSource("predicted-flood-water");
    }
  } catch (error) {
    console.warn(error);
  }

  if (!predictedGeoJSON || predictedGeoJSON.features.length === 0) {
    console.log("No downhill flood expansion detected.");
    return;
  }

  map.addSource("predicted-flood-water", {
    type: "geojson",
    data: predictedGeoJSON,
  });

  map.addLayer({
    id: "predicted-flood-water",
    type: "fill-extrusion",
    source: "predicted-flood-water",
    paint: {
      "fill-extrusion-color": "#36c7f4",
      "fill-extrusion-opacity": 0.6,
      "fill-extrusion-height": 2,
      "fill-extrusion-base": 0,
      "fill-extrusion-vertical-gradient": true,
    },
  });

  console.log("Predicted downhill 3D flood added.");
}

// ---------------------------------------------------------
// POPULATION EXPOSURE ZONES
// ---------------------------------------------------------

function cellKey(x, y) {
  return `${x}_${y}`;
}

// Split the predicted downhill cells into connected flood areas.
// Each connected area gets one location marker.
function buildPopulationZones(predictedCells) {
  if (!predictedCells || predictedCells.length === 0) {
    return [];
  }

  const cellMap = new Map();

  for (const cell of predictedCells) {
    cellMap.set(cellKey(cell.x, cell.y), cell);
  }

  const visited = new Set();
  const zones = [];

  // Use 4-neighbour connectivity here. 8-neighbour connectivity can make
  // diagonally touching cells part of the same zone, which can create
  // self-touching/self-intersecting GeoJSON polygons for WorldPop.
  const directions = [
    [0, -1],
    [-1, 0],
    [1, 0],
    [0, 1],
  ];

  for (const cell of predictedCells) {
    const startKey = cellKey(cell.x, cell.y);

    if (visited.has(startKey)) continue;

    const queue = [cell];
    visited.add(startKey);
    const component = [];

    while (queue.length > 0) {
      const current = queue.shift();
      component.push(current);

      for (const [dx, dy] of directions) {
        const nx = current.x + dx;
        const ny = current.y + dy;
        const k = cellKey(nx, ny);

        if (visited.has(k)) continue;

        const neighbour = cellMap.get(k);
        if (!neighbour) continue;

        visited.add(k);
        queue.push(neighbour);
      }
    }

    if (component.length >= MIN_POPULATION_ZONE_CELLS) {
      zones.push(component);
    }
  }

  // Largest zones first.
  zones.sort((a, b) => b.length - a.length);

  return zones.slice(0, MAX_POPULATION_ZONES);
}

function componentToMultiPolygon(component, width, height, bounds) {
  // WorldPop validates GeoJSON geometry strictly. Sending every grid cell as a
  // separate MultiPolygon can create touching/self-intersecting geometry.
  // Instead, dissolve the connected grid cells into their outer boundary rings.
  if (!component || component.length === 0) {
    return {
      type: "Polygon",
      coordinates: [],
    };
  }

  const cells = new Set(component.map((cell) => cellKey(cell.x, cell.y)));
  const edges = [];

  const addEdge = (x1, y1, x2, y2) => {
    edges.push({
      start: `${x1},${y1}`,
      end: `${x2},${y2}`,
      x1,
      y1,
      x2,
      y2,
    });
  };

  // Add only exposed cell edges. The orientation makes each boundary a
  // closed directed ring.
  for (const cell of component) {
    const { x, y } = cell;

    if (!cells.has(cellKey(x, y - 1))) {
      addEdge(x, y, x + 1, y); // top
    }
    if (!cells.has(cellKey(x + 1, y))) {
      addEdge(x + 1, y, x + 1, y + 1); // right
    }
    if (!cells.has(cellKey(x, y + 1))) {
      addEdge(x + 1, y + 1, x, y + 1); // bottom
    }
    if (!cells.has(cellKey(x - 1, y))) {
      addEdge(x, y + 1, x, y); // left
    }
  }

  const outgoing = new Map();
  for (const edge of edges) {
    if (!outgoing.has(edge.start)) {
      outgoing.set(edge.start, []);
    }
    outgoing.get(edge.start).push(edge);
  }

  const used = new Set();
  const rings = [];

  for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
    if (used.has(edgeIndex)) continue;

    const first = edges[edgeIndex];
    const ring = [[first.x1, first.y1]];
    let current = first;
    used.add(edgeIndex);

    while (true) {
      ring.push([current.x2, current.y2]);

      const nextCandidates = outgoing.get(current.end) || [];
      let nextIndex = -1;

      // Prefer an unused edge. At normal grid boundary vertices there is one.
      for (const candidate of nextCandidates) {
        const idx = edges.indexOf(candidate);
        if (!used.has(idx)) {
          nextIndex = idx;
          break;
        }
      }

      if (nextIndex === -1) break;

      used.add(nextIndex);
      current = edges[nextIndex];

      if (current.end === first.start) {
        ring.push([current.x2, current.y2]);
        break;
      }
    }

    if (ring.length >= 4) {
      // Convert grid coordinates to geographic coordinates.
      const geoRing = ring.map(([gx, gy]) => {
        const longitude =
          bounds.west + (gx / width) * (bounds.east - bounds.west);
        const latitude =
          bounds.north - (gy / height) * (bounds.north - bounds.south);

        return [longitude, latitude];
      });

      // Remove accidental consecutive duplicate points.
      const cleaned = [];
      for (const point of geoRing) {
        const previous = cleaned[cleaned.length - 1];
        if (!previous || previous[0] !== point[0] || previous[1] !== point[1]) {
          cleaned.push(point);
        }
      }

      if (
        cleaned.length >= 4 &&
        cleaned[0][0] === cleaned[cleaned.length - 1][0] &&
        cleaned[0][1] === cleaned[cleaned.length - 1][1]
      ) {
        cleaned.pop();
        cleaned.push([...cleaned[0]]);
      }

      if (cleaned.length >= 4) {
        rings.push(cleaned);
      }
    }
  }

  if (rings.length === 0) {
    throw new Error("Could not construct a valid flood-area polygon.");
  }

  // Calculate signed geographic area. The largest absolute ring is treated
  // as the exterior ring; the remaining rings are holes if present.
  const ringArea = (ring) => {
    let area = 0;
    for (let i = 0; i < ring.length - 1; i++) {
      area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    }
    return area / 2;
  };

  rings.sort((a, b) => Math.abs(ringArea(b)) - Math.abs(ringArea(a)));

  const exterior = rings[0];
  const holes = rings.slice(1);

  return {
    type: "Polygon",
    coordinates: [exterior, ...holes],
  };
}

function componentCenter(component, width, height, bounds) {
  let totalX = 0;
  let totalY = 0;

  for (const cell of component) {
    totalX += cell.x + 0.5;
    totalY += cell.y + 0.5;
  }

  const avgX = totalX / component.length;
  const avgY = totalY / component.length;

  return {
    longitude: bounds.west + (avgX / width) * (bounds.east - bounds.west),
    latitude: bounds.north - (avgY / height) * (bounds.north - bounds.south),
  };
}

async function getWorldPopPopulation(geometry) {
  try {
    console.log("Requesting WorldPop population estimate...");

    const response = await fetch(`${WORLDPOP_API}/population`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        geojson: geometry,
        year: WORLDPOP_YEAR,
        resolution: WORLDPOP_RESOLUTION,
      }),
    });

    if (!response.ok) {
      throw new Error(`WorldPop request failed: HTTP ${response.status}`);
    }

    const submitted = await response.json();
    const taskId = submitted.task_id || submitted.taskid;

    if (!taskId) {
      throw new Error("WorldPop did not return a task ID.");
    }

    console.log("WorldPop task:", taskId);

    for (let attempt = 0; attempt < 30; attempt++) {
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const taskResponse = await fetch(`${WORLDPOP_API}/tasks/${taskId}`);

      if (!taskResponse.ok) {
        throw new Error(
          `WorldPop task request failed: HTTP ${taskResponse.status}`,
        );
      }

      const task = await taskResponse.json();
      const status = String(task.status || "").toLowerCase();

      if (
        status === "success" ||
        status === "finished" ||
        status === "completed"
      ) {
        const population =
          task.result?.total_population ??
          task.data?.total_population ??
          task.total_population;

        if (population === undefined || population === null) {
          throw new Error(
            "WorldPop completed but no population value was returned.",
          );
        }

        return Number(population);
      }

      if (status === "failure" || status === "failed" || task.error === true) {
        throw new Error(
          task.error_message || task.error || "WorldPop task failed.",
        );
      }
    }

    throw new Error("WorldPop request timed out.");
  } catch (error) {
    console.error("WorldPop population error:", error);
    return null;
  }
}

function formatPopulation(value) {
  if (value === null || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return Math.round(value).toLocaleString("en-IN");
}

function clearPopulationMarkers(markerRef) {
  for (const marker of markerRef.current) {
    marker.remove();
  }

  markerRef.current = [];
}

async function calculatePopulationForZones(predictedFlood) {
  if (!predictedFlood?.predictedCells?.length) return [];

  // Copy cells so WorldPop processing never changes flood cells.
  const cells = predictedFlood.predictedCells.map((c) => ({ ...c }));
  const zones = buildPopulationZones(cells);
  const results = [];

  for (let index = 0; index < zones.length; index++) {
    const component = zones[index];
    try {
      const center = componentCenter(
        component,
        predictedFlood.width,
        predictedFlood.height,
        predictedFlood.bounds,
      );
      const geometry = componentToMultiPolygon(
        component,
        predictedFlood.width,
        predictedFlood.height,
        predictedFlood.bounds,
      );
      console.log(`Getting WorldPop population for Flood Area ${index + 1}...`);
      const population = await getWorldPopPopulation(geometry);
      results.push({
        originalIndex: index,
        component,
        center,
        geometry,
        population,
        floodCells: component.length,
        rank: null,
      });
      console.log(`Flood Area ${index + 1}: population =`, population);
    } catch (error) {
      console.error(`WorldPop failed for Flood Area ${index + 1}:`, error);
      results.push({
        originalIndex: index,
        component,
        center: componentCenter(
          component,
          predictedFlood.width,
          predictedFlood.height,
          predictedFlood.bounds,
        ),
        geometry: null,
        population: null,
        floodCells: component.length,
        rank: null,
      });
    }
  }
  return results;
}

function rankPopulationZones(results) {
  const valid = results.filter(
    (z) =>
      z.population !== null &&
      Number.isFinite(z.population) &&
      z.population > 0,
  );
  for (const z of results) z.rank = null;
  valid.sort((a, b) => b.population - a.population);
  let rank = 0,
    previous = null;
  for (const z of valid) {
    if (z.population !== previous) {
      rank++;
      previous = z.population;
    }
    z.rank = rank;
    console.log(
      `Rank ${rank}: Flood Area ${z.originalIndex + 1} -> ${z.population}`,
    );
  }
  return results;
}

function showPopulationRankMarkers(map, markerRef, results) {
  clearPopulationMarkers(markerRef);
  for (const zone of results.filter(
    (z) =>
      z.population !== null &&
      Number.isFinite(z.population) &&
      z.population > 0 &&
      z.rank !== null,
  )) {
    const el = document.createElement("div");
    Object.assign(el.style, {
      width: "38px",
      height: "38px",
      borderRadius: "50%",
      background: "#d7191c",
      border: "3px solid white",
      boxShadow: "0 2px 8px rgba(0,0,0,0.45)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "white",
      fontWeight: "bold",
      fontSize: "17px",
      cursor: "pointer",
    });
    el.innerText = zone.rank;
    const marker = new maplibregl.Marker({ element: el })
      .setLngLat([zone.center.longitude, zone.center.latitude])
      .addTo(map);
    markerRef.current.push(marker);
    const popup = new maplibregl.Popup({
      closeButton: true,
      closeOnClick: true,
      maxWidth: "300px",
    }).setHTML(
      `<div style="font-family:Arial,sans-serif;line-height:1.5;padding:4px;"><div style="font-size:18px;font-weight:bold;margin-bottom:8px;">Flood Area ${zone.originalIndex + 1}</div><div style="font-size:22px;font-weight:bold;margin-bottom:8px;">👥 ${Math.round(zone.population).toLocaleString("en-IN")}</div><div><b>Estimated population exposed</b></div><div style="margin-top:6px;"><b>Population Rank:</b> #${zone.rank}</div><div><b>Flood cells:</b> ${zone.floodCells}</div><div style="margin-top:10px;font-size:11px;color:#666;">Population estimate from WorldPop ${WORLDPOP_YEAR}, ${WORLDPOP_RESOLUTION} data.</div></div>`,
    );
    marker.setPopup(popup);
  }
}

// ---------------------------------------------------------
// GET PREDICTION URL
// ---------------------------------------------------------

function getPredictionUrl(result) {
  if (!result) {
    return null;
  }

  const possibleKeys = [
    "prediction_image",
    "prediction",
    "mask_url",
    "image_url",
    "overlay",
    "overlay_url",
    "result_image",
    "mask",
    "flood_mask",
  ];

  let url = null;

  for (const key of possibleKeys) {
    if (result[key]) {
      url = result[key];

      break;
    }
  }

  if (!url) {
    return null;
  }

  // Backend may return a Base64 data URL.
  // IMPORTANT: do not prepend BACKEND_URL to it.
  if (url.startsWith("data:image/")) {
    console.log("Prediction is a Base64 data URL.");
    return url;
  }

  // Backend sometimes returns:
  // /prediction/abc.png
  if (url.startsWith("/")) {
    return BACKEND_URL + url;
  }

  // Backend may return:
  // prediction/abc.png
  if (!url.startsWith("http")) {
    return BACKEND_URL + "/" + url;
  }

  return url;
}

// ---------------------------------------------------------
// MAP COMPONENT
// ---------------------------------------------------------

function FloodMap() {
  const mapContainer = useRef(null);

  const mapRef = useRef(null);

  // Location markers showing estimated population exposed in each
  // predicted downhill flood area.
  const populationMarkersRef = useRef([]);
  const floodDataRef = useRef(null);
  const predictedFloodRef = useRef(null);
  const populationResultsRef = useRef([]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Step 1: Upload an image");
  const [hasFloodMap, setHasFloodMap] = useState(false);
  const [hasPredictedCells, setHasPredictedCells] = useState(false);
  const [hasPopulationData, setHasPopulationData] = useState(false);

  // -------------------------------------------------------
  // INITIALIZE MAP
  // -------------------------------------------------------

  useEffect(() => {
    if (!mapContainer.current) {
      return;
    }

    if (mapRef.current) {
      return;
    }

    console.log("Initializing 3D Map...");

    const map = new maplibregl.Map({
      container: mapContainer.current,

      style: MAP_STYLE,

      // Initial location
      center: [76.94, 8.52],

      zoom: 10,

      pitch: 55,

      bearing: -20,

      maxPitch: 85,

      antialias: true,
    });

    mapRef.current = map;

    map.addControl(
      new maplibregl.NavigationControl({
        visualizePitch: true,
      }),
      "top-right",
    );

    // -----------------------------------------------------
    // MAP LOAD
    // -----------------------------------------------------

    map.on("load", () => {
      console.log("Map loaded.");

      // -------------------------------------------------
      // TERRAIN
      // -------------------------------------------------

      map.addSource("terrain", {
        type: "raster-dem",

        url: TERRAIN_URL,

        tileSize: 512,

        maxzoom: 12,
      });

      map.setTerrain({
        source: "terrain",
        exaggeration: 1.5,
      });

      // -------------------------------------------------
      // HILLSHADE
      // -------------------------------------------------

      map.addLayer({
        id: "terrain-hillshade",

        type: "hillshade",

        source: "terrain",

        paint: {
          "hillshade-exaggeration": 0.6,
        },
      });

      // -------------------------------------------------
      // 3D BUILDINGS
      // -------------------------------------------------

      map.addSource("buildings", {
        type: "vector",

        url: "https://tiles.openfreemap.org/planet",
      });

      map.addLayer({
        id: "3d-buildings",

        type: "fill-extrusion",

        source: "buildings",

        "source-layer": "building",

        minzoom: 13,

        paint: {
          "fill-extrusion-color": "#888888",

          "fill-extrusion-height": [
            "interpolate",
            ["linear"],
            ["zoom"],

            13,
            0,

            15,
            ["coalesce", ["get", "render_height"], 6],
          ],

          "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],

          "fill-extrusion-opacity": 0.9,
        },
      });

      console.log("3D buildings enabled.");
    });

    map.on("error", (event) => {
      console.warn("MapLibre:", event.error);
    });

    return () => {
      clearPopulationMarkers(populationMarkersRef);

      map.remove();

      mapRef.current = null;
    };
  }, []);

  // -------------------------------------------------------
  // UPLOAD HANDLER
  // -------------------------------------------------------

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setProcessing(true);
    setStatusMessage("Processing Step 1: current flood...");
    console.clear();
    try {
      const csvResponse = await fetch(CSV_FILE);
      if (!csvResponse.ok) throw new Error("Could not load CSV.");
      const rows = parseCSV(await csvResponse.text());
      const location = findImageLocation(rows, file.name);
      if (!location)
        throw new Error("Could not find coordinates for " + file.name);
      const { latitude, longitude } = location;
      const map = mapRef.current;
      if (!map) throw new Error("Map is not initialized.");
      removeFloodLayer(map);
      removePredictedFloodLayer(map);
      clearPopulationMarkers(populationMarkersRef);
      floodDataRef.current = null;
      predictedFloodRef.current = null;
      populationResultsRef.current = [];
      setHasFloodMap(false);
      setHasPredictedCells(false);
      setHasPopulationData(false);
      map.flyTo({
        center: [longitude, latitude],
        zoom: 15,
        pitch: 60,
        bearing: -20,
        duration: 2000,
        essential: true,
      });
      const formData = new FormData();
      formData.append("file", file);
      formData.append("latitude", String(latitude));
      formData.append("longitude", String(longitude));
      const response = await fetch(`${BACKEND_URL}/segment`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok)
        throw new Error("Backend error: " + (await response.text()));
      const result = await response.json();
      const predictionUrl = getPredictionUrl(result);
      if (!predictionUrl)
        throw new Error("Backend did not return a prediction image.");
      const floodData = await predictionToGeoJSON(
        predictionUrl,
        latitude,
        longitude,
      );
      if (!floodData)
        throw new Error("No flood region detected in prediction.");
      floodDataRef.current = floodData;
      map.addSource("flood-water", {
        type: "geojson",
        data: floodData.geoJSON,
      });
      map.addLayer({
        id: "flood-water",
        type: "fill-extrusion",
        source: "flood-water",
        paint: {
          "fill-extrusion-color": "#168bd0",
          "fill-extrusion-opacity": 0.75,
          "fill-extrusion-height": 2,
          "fill-extrusion-base": 0,
          "fill-extrusion-vertical-gradient": true,
        },
      });
      const bounds = new maplibregl.LngLatBounds();
      for (const f of floodData.geoJSON.features)
        for (const ring of f.geometry.coordinates)
          for (const c of ring) bounds.extend(c);
      if (!bounds.isEmpty())
        map.fitBounds(bounds, {
          padding: 100,
          maxZoom: 17,
          pitch: 60,
          bearing: -20,
          duration: 1800,
          essential: true,
        });
      setHasFloodMap(true);
      setStatusMessage("Step 1 complete — current flood map is ready.");
      console.log("STEP 1 COMPLETE — press Predict Flood Cells.");
    } catch (error) {
      console.error("STEP 1 ERROR:", error);
      alert(error.message);
      setStatusMessage("Step 1 failed.");
    } finally {
      setProcessing(false);
    }
  }

  async function handlePredictCells() {
    const map = mapRef.current,
      floodData = floodDataRef.current;
    if (!map || !floodData) return alert("Please upload an image first.");
    setProcessing(true);
    setStatusMessage("Processing Step 2: predicting flood cells...");
    try {
      let first = null;
      for (let y = 0; y < floodData.height && !first; y++)
        for (let x = 0; x < floodData.width; x++)
          if (floodData.floodGrid[y][x]) {
            first = { x, y };
            break;
          }
      if (!first) throw new Error("No flood cells found.");
      const c = getCellCenter(
        first.x,
        first.y,
        floodData.width,
        floodData.height,
        floodData.bounds,
      );
      if (!(await waitForTerrain(map, c.longitude, c.latitude)))
        throw new Error("Terrain DEM is not available.");
      const predicted = await simulateFloodFlow(map, floodData);
      predictedFloodRef.current = predicted;
      addPredictedFloodLayer(map, predicted.geoJSON);
      clearPopulationMarkers(populationMarkersRef);
      populationResultsRef.current = [];
      setHasPredictedCells(true);
      setHasPopulationData(false);
      setStatusMessage(
        `Step 2 complete — ${predicted.predictedCells.length} cells predicted.`,
      );
      console.log("STEP 2 COMPLETE", predicted.predictedCells.length);
    } catch (error) {
      console.error("STEP 2 ERROR:", error);
      alert(error.message);
      setStatusMessage("Step 2 failed.");
    } finally {
      setProcessing(false);
    }
  }

  async function handleWorldPop() {
    const predicted = predictedFloodRef.current;
    if (!predicted) return alert("Please predict flood cells first.");
    setProcessing(true);
    setStatusMessage("Processing Step 3: predicting WorldPop population...");
    try {
      const results = await calculatePopulationForZones(predicted);
      populationResultsRef.current = results;
      clearPopulationMarkers(populationMarkersRef);
      const count = results.filter(
        (z) =>
          z.population !== null &&
          Number.isFinite(z.population) &&
          z.population > 0,
      ).length;
      setHasPopulationData(true);
      setStatusMessage(
        `Step 3 complete — population found for ${count} areas.`,
      );
      console.log("STEP 3 COMPLETE — no rank markers shown yet.");
    } catch (error) {
      console.error("STEP 3 ERROR:", error);
      alert(error.message);
      setStatusMessage("Step 3 failed.");
    } finally {
      setProcessing(false);
    }
  }

  function handleRankPopulation() {
    const map = mapRef.current,
      results = populationResultsRef.current;
    if (!map || !results.length)
      return alert("Please predict WorldPop population first.");
    const ranked = rankPopulationZones(results);
    populationResultsRef.current = ranked;
    showPopulationRankMarkers(map, populationMarkersRef, ranked);
    setStatusMessage("Step 4 complete — areas ranked by population.");
    console.log("STEP 4 COMPLETE — red markers show population rank.");
  }

  // -------------------------------------------------------
  // UI
  // -------------------------------------------------------

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />

      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          zIndex: 10,
          background: "rgba(255,255,255,0.96)",
          padding: "16px",
          borderRadius: "10px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.2)",
          width: "270px",
        }}
      >
        <div
          style={{ fontWeight: "bold", fontSize: "18px", marginBottom: "12px" }}
        >
          Flood Impact Assessment
        </div>

        <div
          style={{ fontSize: "12px", fontWeight: "bold", marginBottom: "5px" }}
        >
          STEP 1 — CURRENT FLOOD
        </div>
        <input
          type="file"
          accept="image/*"
          onChange={handleUpload}
          disabled={processing}
          style={{ width: "100%" }}
        />

        <button
          onClick={handlePredictCells}
          disabled={!hasFloodMap || processing}
          style={{ width: "100%", padding: "9px", marginTop: "9px" }}
        >
          2. Predict Flood Cells
        </button>
        <button
          onClick={handleWorldPop}
          disabled={!hasPredictedCells || processing}
          style={{ width: "100%", padding: "9px", marginTop: "7px" }}
        >
          3. Predict WorldPop
        </button>
        <button
          onClick={handleRankPopulation}
          disabled={!hasPopulationData || processing}
          style={{ width: "100%", padding: "9px", marginTop: "7px" }}
        >
          4. Rank Population
        </button>

        <div
          style={{
            marginTop: "12px",
            fontSize: "13px",
            color: processing ? "#1565c0" : "#444",
            lineHeight: 1.4,
          }}
        >
          {statusMessage}
        </div>
        {selectedFile && (
          <div
            style={{
              marginTop: "8px",
              fontSize: "11px",
              color: "#777",
              wordBreak: "break-word",
            }}
          >
            File: {selectedFile.name}
          </div>
        )}
      </div>
    </div>
  );
}
// ---------------------------------------------------------
// DEFAULT EXPORT
// ---------------------------------------------------------

export default FloodMap;
