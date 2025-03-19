// Global variables
let map;
let drawingManager;
let currentShape = null;
let regions = [];
let markers = [];
let drawingHistory = []; // For undo functionality


// Sector colors mapping
const sectorColors = {
    "oil-and-gas-transport": "#1e88e5",
    "oil-and-gas-production": "#e53935",
    "net-shrubgrass": "#8d6e63",
    "international-shipping": "#fdd835",
    "net-forest-land": "#43a047",
    "iron-and-steel": "#757575",
    "electricity-generation": "#1565c0",
    "road-transportation": "#ff5722",
    "chemicals": "#5e35b1",
    "residential-onsite-fuel-usage": "#8e24aa",
    "oil-and-gas-refining": "#00897b",
    "international-aviation": "#d81b60",
    "aluminum": "#039be5",
    "lime": "#26a69a",
    "cement": "#7cb342",
    "coal-mining": "#8d4004",
    "other-chemicals": "#6a1b9a",
    "non-residential-onsite-fuel-usage": "#f4511e",
    "petrochemical-steam-cracking": "#ec407a",
    "rice-cultivation": "#795548",
    "solid-waste-disposal": "#9e9e9e",
    "domestic-shipping": "#4fc3f7",
    "domestic-aviation": "#558b2f",
    "enteric-fermentation-cattle-pasture": "#ffb74d",
    "net-wetland": "#00838f",
    "cropland-fires": "#b8860b",
    "food-beverage-tobacco": "#2e7d32",
    "copper-mining": "#9e9d24",
    "water-reservoirs": "#2e7c67",
    "iron-mining": "#4527a0",
    "manure-left-on-pasture-cattle": "#6a0dad",
    "synthetic-fertilizer-application": "#d84315",
    "bauxite-mining": "#c2185b",
    "pulp-and-paper": "#00acc1",
    "glass": "#1976d2"
};

// Initialize map and drawing tools
function initMap() {
    // Create map instance
    console.log("Map is initializing...");
    map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 22.3511, lng: 78.6677 },
        zoom: 5,
        mapTypeControl: true,
        mapTypeControlOptions: {
            style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
            mapTypeIds: ['roadmap', 'satellite']
        },
        styles: getMapStyle('roadmap')
    });

    // Add close button event listener
    document.getElementById('closeEmissions').addEventListener('click', () => {
        document.getElementById('emissionsData').classList.add('hidden');
        map.setZoom(5); // Reset to initial zoom level
        map.setCenter({ lat: 22.3511, lng: 78.6677 }); // Reset to initial center
    });

    // Load GeoJSON (only call this once)
    loadGeoJson();

let allFeatures = []; // Store all features globally

// Load GeoJSON data and update dynamically
async function loadGeoJson() {
    try {
        const response = await fetch("/maps1/static/data.geojson"); // Updated path to match static file configuration
        const geojsonData = await response.json(); // Convert to JSON
        console.log("Loaded GeoJSON Data:", geojsonData); // Debugging output

        // Store all features globally
        allFeatures = geojsonData.features;

        // Initially, show features based on zoom level
        updateMapData();

        // Listen for zoom changes
        map.addListener('zoom_changed', updateMapData);

    } catch (error) {
        console.error('Error loading GeoJSON:', error);
    }
}

// Function to update the map based on zoom level
function updateMapData() {
    const zoomLevel = map.getZoom();
    console.log("Current Zoom Level:", zoomLevel);

    // Clear existing map data
    map.data.forEach(feature => {
        map.data.remove(feature);
    });

    let selectedFeatures = allFeatures;

    if (zoomLevel >= 10) { // Show only 100 points at zoom level 10 or higher
        selectedFeatures = getRandomSubset(allFeatures, 500);
    } else if(zoomLevel >= 7) { // Show only 50 points at zoom level 7-9
        selectedFeatures = getRandomSubset(allFeatures, 400);
    } else { // Show only 20 points for lower zoom
        selectedFeatures = getRandomSubset(allFeatures, 600);
    }

    // Add filtered data back to the map
    map.data.addGeoJson({ type: 'FeatureCollection', features: selectedFeatures });

    // Apply styling
    map.data.setStyle(feature => ({
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: sectorColors[feature.getProperty('Sector')] || '#808080',
            fillOpacity: 1,
            strokeWeight: 2,
            strokeColor: '#fff',
            scale: 8
        }
    }));
}

// Function to get a random subset of features
function getRandomSubset(array, num) {
    return array.sort(() => 0.5 - Math.random()).slice(0, num);
}

// Handle clicks on GeoJSON features
// Handle clicks on GeoJSON features
map.data.addListener('click', async (event) => {
    const feature = event.feature;
    const name = feature.getProperty('Name');
    const sector = feature.getProperty('Sector');
    const assetId = feature.getProperty('sourceID'); // Assuming this is stored in your GeoJSON

    map.setCenter(event.latLng);
    map.setZoom(14);

    // Show loading state in the emissions box
    document.getElementById("regionName").textContent = name;
    document.getElementById("sectorType").textContent = sector;
    document.getElementById("carbonEmissions").textContent = "Loading...";
    document.getElementById("methaneEmissions").textContent = "Loading...";
    document.getElementById("nitrousEmissions").textContent = "Loading...";
    document.getElementById("emissionsData").classList.remove("hidden");

    if (!assetId) {
        console.warn("No asset ID found for this feature.");
        return;
    }

    // Fetch emissions data
    try {
        const url = `https://api.climatetrace.org/v6/assets/${assetId}`;
        const options = { method: 'GET' };
        
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();
        console.log("Fetched emissions data:", data);

        // Extract emissions quantities from the response
        let carbonEmissions = "N/A";
        let methaneEmissions = "N/A";
        let nitrousEmissions = "N/A";

        // Find emissions details for the gases
        data.EmissionsSummary.forEach(emission => {
            switch (emission.Gas) {
                case 'co2':
                    carbonEmissions = emission.EmissionsQuantity;
                    break;
                case 'ch4':
                    methaneEmissions = emission.EmissionsQuantity;
                    break;
                case 'n2o':
                    nitrousEmissions = emission.EmissionsQuantity;
                    break;
            }
        });

        // Update the UI with emissions data
        document.getElementById("carbonEmissions").textContent = carbonEmissions;
        document.getElementById("methaneEmissions").textContent = methaneEmissions;
        document.getElementById("nitrousEmissions").textContent = nitrousEmissions;

    } catch (error) {
        console.error("Error fetching emissions data:", error);
        document.getElementById("carbonEmissions").textContent = "Error";
        document.getElementById("methaneEmissions").textContent = "Error";
        document.getElementById("nitrousEmissions").textContent = "Error";
    }
});
    // Add map type change listener
    map.addListener('maptypeid_changed', () => {
        currentMapType = map.getMapTypeId();
        map.setOptions({ styles: getMapStyle(currentMapType) });
        updateMarkerLabels();
    });

    // Initialize Climate Trace data layer
    loadClimateTraceData();

    // Initialize drawing manager
    drawingManager = new google.maps.drawing.DrawingManager({
        drawingMode: null,
        drawingControl: false,
        polygonOptions: {
            fillColor: '#FF0000',
            fillOpacity: 0.3,
            strokeWeight: 2,
            clickable: true,
            editable: true,
            zIndex: 1
        }
    });

    // Initialize search with autocomplete
    const searchInput = document.getElementById('mapSearch');
    const autocomplete = new google.maps.places.Autocomplete(searchInput, {
        types: ['geocode']
    });

    autocomplete.bindTo('bounds', map);
    
    autocomplete.addListener('place_changed', () => {
        const place = autocomplete.getPlace();
        
        if (!place.geometry) {
            console.warn("Returned place contains no geometry");
            return;
        }

        if (place.geometry.viewport) {
            map.fitBounds(place.geometry.viewport);
        } else {
            map.setCenter(place.geometry.location);
            map.setZoom(17);
        }
    });

    // Set up event listeners
    setupEventListeners();
    
    // Generate initial markers
    generateRandomMarkers(100);
}

function generateRandomMarkers(count) {
    markers = []; // Clear old markers

    allFeatures.forEach(feature => {
        if (feature.geometry && feature.geometry.type === "Point") {
            createMarker(feature, {
                lat: feature.geometry.coordinates[1],
                lng: feature.geometry.coordinates[0]
            });
        }
    });

    updateMarkerClustering(); // Apply clustering after markers are created
}

function updateMarkerClustering() {
    if (markerCluster) {
        markerCluster.clearMarkers(); // Clear previous clusters
    }

    markerCluster = new MarkerClusterer(map, markers, {
        imagePath: 'https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/m',
        gridSize: 50, // Adjust for performance (higher means more clustering)
        maxZoom: 12   // Prevents clustering at zoom level 12 or higher
    });
}



// Event listeners setup
function setupEventListeners() {
    const addAssetBtn = document.getElementById('addAsset');
    const cancelDrawBtn = document.getElementById('cancelDraw');
    const saveDrawBtn = document.getElementById('saveDraw');
    const undoBtn = document.getElementById('undoDraw');
    const regionSearch = document.getElementById('regionSearch');
    const modalOverlay = document.getElementById('modalOverlay');
    const closeEmissionsBtn = document.getElementById('closeEmissions');

    addAssetBtn.addEventListener('click', startDrawing);
    cancelDrawBtn.addEventListener('click', cancelDrawing);
    saveDrawBtn.addEventListener('click', saveShape);
    undoBtn.addEventListener('click', undoLastPoint);
    regionSearch.addEventListener('input', filterRegions);
    modalOverlay.addEventListener('click', closeModal);
    closeEmissionsBtn.addEventListener('click', closeEmissionsData);

    google.maps.event.addListener(drawingManager, 'polygoncomplete', (polygon) => {
        currentShape = polygon;
        document.getElementById('saveDraw').classList.remove('hidden');
        document.getElementById('undoDraw').classList.remove('hidden');
        
        // Store points for undo functionality
        const path = polygon.getPath();
        path.addListener('insert_at', () => {
            drawingHistory.push([...path.getArray()]);
        });
    });

    // Generate initial markers immediately
    setTimeout(() => {
        generateRandomMarkers(200);
    }, 1000);
}

// Undo last point in polygon
function undoLastPoint() {
    if (!currentShape) return;
    
    const path = currentShape.getPath();
    if (path.getLength() > 0) {
        path.removeAt(path.getLength() - 1);
        drawingHistory.pop();
    }
    
    if (path.getLength() === 0) {
        cancelDrawing();
    }
}

// Create a marker on the map
let markerCluster; // Declare marker cluster variable
function createMarker(feature, position) {
    const color = sectorColors[feature.properties.Sector] || '#808080';
    const marker = new google.maps.Marker({
        position: position,
        map: map,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: color,
            fillOpacity: 1,
            strokeWeight: 2,
            strokeColor: '#FFFFFF',
            scale: 6
        }
    });

    //marker.addListener('click', () => {
      //  map.setZoom(14);
        //map.setCenter(marker.getPosition());
       // showMoreDetails(feature);
    //});

    markers.push(marker);
}

// Clear all markers from the map
function clearMarkers() {
    markers.forEach(marker => marker.setMap(null));
    markers = [];
}

// Start drawing mode
function startDrawing() {
    drawingManager.setMap(map);
    drawingManager.setDrawingMode(google.maps.drawing.OverlayType.POLYGON);
    document.getElementById('addAsset').classList.add('hidden');
    document.getElementById('cancelDraw').classList.remove('hidden');
    document.getElementById('saveDraw').classList.remove('hidden');
    document.getElementById('undoDraw').classList.remove('hidden');
    drawingHistory = [];
}

// Cancel drawing mode
function cancelDrawing() {
    if (currentShape) {
        currentShape.setMap(null);
        currentShape = null;
    }
    resetDrawingState();
}

// Reset drawing state
function resetDrawingState() {
    drawingManager.setMap(null);
    drawingManager.setDrawingMode(null);
    document.getElementById('addAsset').classList.remove('hidden');
    document.getElementById('cancelDraw').classList.add('hidden');
    document.getElementById('saveDraw').classList.add('hidden');
    document.getElementById('undoDraw').classList.add('hidden');
    drawingHistory = [];
}

// Show details modal
function showMoreDetails(feature) {
    console.log(feature)
    const data = {
        Name: feature.properties.Name,
        Sector: feature.properties.Sector,
        EmissionsDetails: [
            { Gas: 'Carbon Dioxide', EmissionsQuantity: feature.properties.emissions.carbon },
        ]
    };
    displayModal(data);
}

// Display modal with data
function displayModal(data) {
    console.log("EmissionsDetails:", data.EmissionsDetails); // Debugging output
    const modalContent = document.getElementById('modalContent');
    const emissionYear = data.Year && data.Year !== "N/A" ? `Year: ${data.Year}` : "Year: Not Available";

    modalContent.innerHTML = `
        <h4 class="modal-title">${data.Name}</h4>
        <div class="sector-badge" style="background-color: ${sectorColors[data.Sector] || '#808080'}">
            ${data.Sector}
        </div>
        <p><strong>${emissionYear}</strong></p>
        <div class="emissions-details">
            ${data.EmissionsDetails.map(e => `
                <div class="emission-item">
                    <strong>${e.Gas.toUpperCase()}:</strong> 
                    <span>${formatNumber(e.EmissionsQuantity)} metric tons</span>
                </div>
            `).join('')}
        </div>
    `;

    document.getElementById('detailsModal').style.display = 'block';
    document.getElementById('modalOverlay').style.display = 'block';
}

// Close modal
function closeModal() {
    document.getElementById('detailsModal').style.display = 'none';
    document.getElementById('modalOverlay').style.display = 'none';
}

// Close emissions data box
function closeEmissionsData() {
    document.getElementById('emissionsData').classList.add('hidden');
}

// Format numbers with commas
function formatNumber(number) {
    return new Intl.NumberFormat().format(number);
}

// Save drawn shape
// Modified saveShape function
// Save drawn shape and load filtered Climate Trace data based on the drawn region// Helper function to extract the country code from geocoder results
function extractCountryCode(results) {
    for (const result of results) {
        for (const component of result.address_components) {
            if (component.types.includes("country")) {
                return component.short_name; // e.g., "US" or "IN"
            }
        }
    }
    return 'Unknown';
}

// Modified saveShape function with updated reverse geocoding
async function saveShape() {
    if (!currentShape) return;

    // Extract the coordinates from the drawn polygon
    const coordinates = currentShape.getPath().getArray().map(coord => ({
        lat: coord.lat(),
        lng: coord.lng()
    }));

    // Calculate the center of the polygon using getPolygonCenter (already defined)
    const center = getPolygonCenter(coordinates);

    // Calculate the radius (maximum distance from the center to any vertex) using calculatePolygonRadius
    const radiusKm = calculatePolygonRadius(coordinates, center);

    // Use reverse geocoding to get the country code from the center coordinates
    const geocoder = new google.maps.Geocoder();
    geocoder.geocode({ location: center }, async (results, status) => {
        let countryCode = 'IND';
        if (status === 'OK' && results && results.length > 0) {
            // Use the helper function to extract the country code from the results
            console.log("results", results);    
            // countryCode = extractCountryCode(results);
        } else {
            console.warn('Reverse geocoding failed: ', status);
            // countryCode = 'Unknown';
        }

        // Log the calculated values (for debugging)
        console.log("Center Latitude:", center.lat());
        console.log("Center Longitude:", center.lng());
        console.log("Country Code:", countryCode);
        console.log("Radius (km):", radiusKm);

        // Now, use these parameters to load data.
        await loadClimateTraceData({
            countryCode: countryCode,
            searchCoords: { latitude: center.lat(), longitude: center.lng() },
            maxDistanceKm: radiusKm,
            companyName: "" // Optionally, pass a company name filter if needed
        });
        
        // Optionally update your UI here
    });

    // Remove the drawn shape from the map and reset the drawing UI
    currentShape.setMap(null);
    currentShape = null;
    resetDrawingState();
}

// Get polygon center
function getPolygonCenter(coordinates) {
    const bounds = new google.maps.LatLngBounds();
    coordinates.forEach(coord => {
        bounds.extend(new google.maps.LatLng(coord.lat, coord.lng));
    });
    return bounds.getCenter();
}

// Get region name from coordinates
function getRegionName(geocoder, center) {
    return new Promise((resolve, reject) => {
        geocoder.geocode({ location: center }, (results, status) => {
            if (status === 'OK' && results && results.length > 0) {
                resolve(results[0].formatted_address);
            } else {
                resolve('Unknown Region');
            }
        });
    });
}

// Delete region
function deleteRegion(regionId) {
    const regionIndex = regions.findIndex(r => r.id === regionId);
    if (regionIndex === -1) return;

    const region = regions[regionIndex];
    region.shape.setMap(null);
    regions.splice(regionIndex, 1);
    
    const regionElement = document.querySelector(`[data-region-id="${regionId}"]`);
    if (regionElement) {
        regionElement.remove();
    }
    
    closeEmissionsData();
}

// Add region to sidebar list
function addRegionToList(region) {
    const regionsList = document.getElementById('regionsList');
    const regionElement = document.createElement('div');
    regionElement.className = 'region-item';
    regionElement.setAttribute('data-region-id', region.id);
    regionElement.innerHTML = `
        <div class="region-thumbnail" style="background-color: ${sectorColors[region.emissions.sectorType]}"></div>
        <div class="region-info">
            <h4>${region.name}</h4>
            <p>Coordinates: ${region.coordinates[0].lat.toFixed(2)}, ${region.coordinates[0].lng.toFixed(2)}</p>
        </div>
        <button class="delete-region" onclick="deleteRegion(${region.id})">×</button>
    `;

    regionElement.addEventListener('click', (e) => {
        if (!e.target.classList.contains('delete-region')) {
            const center = getPolygonCenter(region.coordinates);
            map.setCenter(center);
            map.setZoom(14);
            displayEmissionsData(region.emissions, region.name);
        }
    });

    regionsList.appendChild(regionElement);
}

// Display emissions data in the box
function displayEmissionsData(data, regionName) {
    const emissionsBox = document.getElementById('emissionsData');
    document.getElementById('regionName').textContent = data.Name;
    document.getElementById('sectorType').textContent = data.Sector || 'Unknown';
    document.getElementById('carbonEmissions').textContent = formatNumber(data.EmissionsQuantity) || '0';
    // document.getElementById('methaneEmissions').textContent = formatNumber(data.methaneEmissions) || '0';
    // document.getElementById('nitrousEmissions').textContent = formatNumber(data.nitrousEmissions) || '0';
    emissionsBox.classList.remove('hidden');
}

// Filter regions in sidebar
function filterRegions(event) {
    const searchTerm = event.target.value.toLowerCase();
    const regionElements = document.querySelectorAll('.region-item');
    
    regionElements.forEach(element => {
        const regionName = element.querySelector('h4').textContent.toLowerCase();
        element.style.display = regionName.includes(searchTerm) ? 'flex' : 'none';
    });
}

// Initialize the map when the page loads
window.onload = initMap;

// Dynamic map styling based on map type
function getMapStyle(mapType) {
    return mapType === 'satellite' ? 
    [
        {
            "elementType": "labels.text.fill",
            "stylers": [{ "color": "#ffffff" }]
        },
        {
            "elementType": "labels.text.stroke",
            "stylers": [{ "color": "#000000" }, { "weight": 2 }]
        }
    ] : 
    [
        {
            "elementType": "labels.text.fill",
            "stylers": [{ "color": "#1a1a1a" }]
        },
        {
            "featureType": "water",
            "elementType": "geometry",
            "stylers": [{ "color": "#e9e9e9" }]
        },
        {
            "featureType": "landscape",
            "elementType": "geometry",
            "stylers": [{ "color": "#f5f5f5" }]
        }
    ];
}
function xhrRequest(url) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("GET", url, true);
        
        xhr.onload = function () {
            if (xhr.status === 200) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                reject(`Error: ${xhr.status}`);
            }
        };
        
        xhr.onerror = () => reject("Request failed");
        xhr.send();
    });
}

async function loadClimateTraceData({ countryCode, searchCoords, maxDistanceKm, companyName }) {
    try {
        const params = new URLSearchParams({
            countries: countryCode || '',
            limit: '5000', // Adjust as needed
            offset: '0'
        });

        const url = `https://api.climatetrace.org/v6/assets?${params.toString()}`;

        const data = await xhrRequest(url);
        console.log("Response Data:", data);
        // const data = await response.json();
        clearMarkers();

        let filteredAssets = data.assets || [];

        if (searchCoords && maxDistanceKm) {
            console.log("1")
            filteredAssets = filteredAssets.filter(asset => {
                if (asset.Centroid?.Geometry) {
                    const [lon, lat] = asset.Centroid.Geometry;
                    const assetCoords = { latitude: lat, longitude: lon };
                    const distance = getDistance(searchCoords, assetCoords);
                    return distance <= maxDistanceKm;
                }
                return false;
            });
            console.log("2")
        } else if (companyName) {
            console.log("3")
            filteredAssets = filteredAssets.filter(asset => 
                asset.Name && asset.Name.toLowerCase().includes(companyName.toLowerCase())
            );
            console.log("4")
        }
console.log("5")
        filteredAssets.forEach(asset => createClimateTraceMarker(asset));
console.log("6")
    } catch (error) {
        console.error('Error loading Climate Trace data:', error);
        alert('Failed to load emissions data. Please try again later.');
    }
}

// Function to calculate the distance between two coordinates (Haversine formula)
function getDistance(coord1, coord2) {
    const R = 6371; // Earth's radius in km
    const dLat = (coord2.latitude - coord1.latitude) * (Math.PI / 180);
    const dLon = (coord2.longitude - coord1.longitude) * (Math.PI / 180);
    const a = 
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(coord1.latitude * (Math.PI / 180)) * 
        Math.cos(coord2.latitude * (Math.PI / 180)) * 
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}
// Helper: Calculate maximum distance (in km) from the center to all vertices of the polygon.
function calculatePolygonRadius(coordinates, center) {
    let maxDistance = 0;
    coordinates.forEach(coord => {
        // Using your existing getDistance function:
        const distance = getDistance(
            { latitude: center.lat(), longitude: center.lng() },
            { latitude: coord.lat, longitude: coord.lng }
        );
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    });
    return maxDistance;
}


// Create marker with Climate Trace data
function createClimateTraceMarker(asset) {
    const position = new google.maps.LatLng(asset.Centroid.Geometry[1], asset.Centroid.Geometry[0]);
    const color = sectorColors[asset.Sector] || '#808080';
    
    const marker = new google.maps.Marker({
        position: position,
        map: map,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: color,
            fillOpacity: 0.8,
            strokeWeight: 2,
            strokeColor: '#000',
            scale: 8
        },
        title: asset.Name
    });

    // When marker is clicked, fetch detailed data
    marker.addListener('click', async () => {
        console.log(`Fetching data for asset ID: ${asset.Id}`);
        
        try {
            const url = `https://api.climatetrace.org/v6/assets/${asset.Id}`;
            const response = await fetch(url);
            
            if (!response.ok) throw new Error(`API request failed: ${response.status}`);
            
            const data = await response.json();
            console.log("Fetched asset details:", data);
            
            // Pass the fetched data to showAssetDetails
            showAssetDetails(data);
        } catch (error) {
            console.error("Error fetching asset details:", error);
            alert("Failed to fetch asset details.");
        }
    });

    markers.push(marker);
}


// Show detailed asset information
//function showAssetDetails(asset) {
    //if (!asset.EmissionsSummary || asset.EmissionsSummary.length === 0) {
      //  alert("No emissions data available for this asset.");
       // return;
    //}

    //let emissionsData = {
      //  Name: asset.Name || "Unknown",
        //Sector: asset.Sector || "Unknown",
        //EmissionsDetails: []
    //};

    //asset.EmissionsSummary.forEach(emission => {
      //  if (emission.EmissionsQuantity > 0) { // Only add gases with emissions > 0
        //    emissionsData.EmissionsDetails.push({
          //      Gas: emission.Gas.toUpperCase(),
            //    EmissionsQuantity: emission.EmissionsQuantity.toFixed(2), // Rounded for UI clarity
            //});
        //}
    //});

    //displayModal(emissionsData);
//}
function showAssetDetails(asset) {
    console.log("Fetched emissions data:", asset);  // ✅ Debugging log

    if (!asset.EmissionsDetails || asset.EmissionsDetails.length === 0) {
        alert("No emissions data available for this asset.");
        return;
    }

    let emissionsData = {
        Name: asset.Name || "Unknown",
        Sector: asset.Sector || "Unknown",
        EmissionsDetails: [],
        Year: "N/A"  // Default value
    };

    // ✅ Extract Year from first available entry in EmissionsDetails
    if (asset.EmissionsDetails.length > 0) {
        emissionsData.Year = asset.EmissionsDetails[0].Year || "N/A";
    }

    // ✅ Use EmissionsDetails instead of EmissionsSummary
    asset.EmissionsDetails.forEach(emission => {
        if (emission.EmissionsQuantity > 0) {
            emissionsData.EmissionsDetails.push({
                Gas: emission.Gas.toUpperCase(),
                EmissionsQuantity: emission.EmissionsQuantity.toFixed(2),
                Year: emission.Year  // ✅ Ensure each gas stores its year
            });
        }
    });

    displayModal(emissionsData);
}


// Update marker labels on map type change
function updateMarkerLabels() {
    markers.forEach(marker => {
        marker.setIcon({
            ...marker.getIcon(),
            strokeColor: currentMapType === 'satellite' ? '#fff' : '#000'
        });
    });
}

// Modified displayModal function for Climate Trace data
function displayModal(data) {
    console.log("EmissionsDetails:", data.EmissionsDetails); // Debugging output
    const modalContent = document.getElementById('modalContent');
    const emissionYear = data.Year && data.Year !== "N/A" ? `Year: ${data.Year}` : "Year: Not Available";

    modalContent.innerHTML = `
        <h4 class="modal-title">${data.Name}</h4>
        <div class="sector-badge" style="background-color: ${sectorColors[data.Sector] || '#808080'}">
            ${data.Sector}
        </div>
        <p><strong>${emissionYear}</strong></p>
        <div class="emissions-details">
            ${data.EmissionsDetails.map(e => `
                <div class="emission-item">
                    <strong>${e.Gas.toUpperCase()}:</strong> 
                    <span>${formatNumber(e.EmissionsQuantity)} metric tons</span>
                </div>
            `).join('')}
        </div>
    `;

    document.getElementById('detailsModal').style.display = 'block';
    document.getElementById('modalOverlay').style.display = 'block';
}

