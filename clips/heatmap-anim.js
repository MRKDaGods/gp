/* ============================================================================
   heatmap-anim.js
   
   Geospatially Constrained Search Animation Engine
   
   This animation simulates a search expanding from a starting camera across
   nearby cameras (within a growing radius) until the next suspect camera is
   found, at which point the radar center moves to that camera and repeats.
   
   ALL CONFIGURATION IS AT THE TOP OF THIS FILE. Change these values to
   customize the animation without touching the logic below.
   
   ============================================================================ */

/* ============================================================================
   CONFIGURATION: All customizable parameters
   ============================================================================ */

const CONFIG = {
    // ========== MAP & DISPLAY ==========
    // Set mapImageUrl to a URL or data:image/png;base64,... string
    // If empty, a simple SVG placeholder will be used
    mapImageUrl: '',

    // Total map dimensions (pixels)
    mapWidth: 800,
    mapHeight: 600,

    // ========== CAMERAS ==========
    // Array of camera objects. Each camera has:
    //   id: unique identifier (1-20)
    //   x, y: pixel position relative to map container
    //   label: string/number to display inside the pin (e.g., "1", "C5")
    //   isSuspectLocation: boolean - is this one of the 4 suspect cameras?
    //   detectionOrder: integer 1-4 (if isSuspectLocation=true); ignored otherwise
    cameras: [
        // Suspect cameras (these are where the suspect is detected)
        { id: 1, x: 100, y: 150, label: '1', isSuspectLocation: false, detectionOrder: 0 },
        { id: 2, x: 180, y: 100, label: '2', isSuspectLocation: true, detectionOrder: 1 },
        { id: 5, x: 350, y: 200, label: '5', isSuspectLocation: true, detectionOrder: 2 },
        { id: 7, x: 500, y: 250, label: '7', isSuspectLocation: true, detectionOrder: 3 },
        { id: 10, x: 600, y: 350, label: '10', isSuspectLocation: true, detectionOrder: 4 },
        { id: 14, x: 700, y: 480, label: '14', isSuspectLocation: true, detectionOrder: 5 },

        // Non-suspect cameras (background cameras being searched)
        { id: 3, x: 250, y: 120, label: '3', isSuspectLocation: false, detectionOrder: 0 },
        { id: 4, x: 320, y: 140, label: '4', isSuspectLocation: false, detectionOrder: 0 },
        { id: 6, x: 450, y: 180, label: '6', isSuspectLocation: false, detectionOrder: 0 },
        { id: 8, x: 520, y: 320, label: '8', isSuspectLocation: false, detectionOrder: 0 },
        { id: 9, x: 580, y: 280, label: '9', isSuspectLocation: false, detectionOrder: 0 },
        { id: 11, x: 650, y: 300, label: '11', isSuspectLocation: false, detectionOrder: 0 },
        { id: 12, x: 720, y: 280, label: '12', isSuspectLocation: false, detectionOrder: 0 },
        { id: 13, x: 750, y: 400, label: '13', isSuspectLocation: false, detectionOrder: 0 },
        { id: 15, x: 300, y: 450, label: '15', isSuspectLocation: false, detectionOrder: 0 },
        { id: 16, x: 550, y: 500, label: '16', isSuspectLocation: false, detectionOrder: 0 },
    ],

    // ========== ANIMATION TIMING ==========
    // Total duration of the entire animation (seconds)
    animationDurationSeconds: 15,

    // Duration of the first phase (from start to first detection)
    // This phase is longer to explain the search logic to viewers
    // Subsequent phases are calculated as (total - first) / (numPhases - 1)
    firstPhaseDurationSeconds: 6,

    // How long a single radar expansion takes (min radius → max radius)
    radarGrowthDurationSeconds: 2.5,

    // Pause after each detection before moving to next camera (seconds)
    detectionPauseDurationSeconds: 0.8,

    // ========== RADAR VISUALIZATION ==========
    // Minimum radius of the radar ring when it starts (pixels)
    radarMinRadiusPx: 20,

    // Maximum radius of the radar ring when it fully expands (pixels)
    radarMaxRadiusPx: 240,

    // Speed multiplier for how fast the radar searches through nearby cameras
    // 1.0 = normal, 2.0 = twice as fast, 0.5 = half as fast
    searchSpeedFactor: 1.0,

    // ========== COLORS & STYLING ==========
    radarColor: '#00ff00',
    inactiveCameraColor: '#64748b',
    searchingCameraColor: '#ff8c00',
    detectedCameraColor: '#10b981',

    radarOpacity: 0.7,
    radarStrokeWidth: 2.5,

    routeLineColor: '#0ea5e9',
    routeLineWidth: 3,
    routeWaypointRadius: 6,

    // ========== TEXT LABELS ==========
    // Show text labels during animation?
    showTextLabels: true,

    // Show the final route line connecting all suspect cameras?
    showRouteLine: true,

    // ========== CAMERA STYLING ==========
    // Scale factor for camera pin size (1.0 = default, 1.5 = 50% larger)
    cameraScale: 1.0,

    // DEBUG: Set to true to log phase timings and camera data
    debugLogging: false,
};

/* ============================================================================
   INTERNAL STATE
   ============================================================================ */

let animationState = {
    isPlaying: false,
    currentTime: 0,
    startTime: null,
    phases: [], // Computed in init()
    detectedCameras: new Set(), // Set of camera IDs that have been detected
    searchedCameras: new Set(), // Set of camera IDs currently being searched
    radarCenterCameraId: null, // Current camera ID at the center of the radar
    currentPhaseIndex: 0, // Which phase (0-3) are we in?
    nextSuspectDetected: false, // Has the next suspect camera entered the search radius?
    nextSuspectDetectionTime: null, // When was the next suspect detected
    firstPhaseCompleteLabelShown: false, // Has the first phase completion label been shown?
    speedConstraintLabelShown: false, // Has the speed constraint text been shown once?
};

/* ============================================================================
   UTILITY FUNCTIONS
   ============================================================================ */

/**
 * Euclidean distance between two points
 */
function distance(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

/**
 * Get camera object by ID
 */
function getCameraById(cameraId) {
    return CONFIG.cameras.find(c => c.id === cameraId);
}

/**
 * Get all cameras within a given radius from a center point
 */
function getCamerasWithinRadius(centerX, centerY, radius) {
    return CONFIG.cameras.filter(camera => {
        const dist = distance(centerX, centerY, camera.x, camera.y);
        return dist <= radius && !animationState.detectedCameras.has(camera.id);
    });
}

/**
 * Get all suspect cameras sorted by detection order
 */
function getSuspectCameras() {
    return CONFIG.cameras
        .filter(c => c.isSuspectLocation)
        .sort((a, b) => a.detectionOrder - b.detectionOrder);
}

/**
 * Calculate the radar max radius needed to reach the next suspect camera
 * Returns distance from current camera to next camera + 50 pixel buffer
 */
function calculateRadarMaxRadius(currentPhaseIndex) {
    const suspectCameras = getSuspectCameras();
    
    // If this is the last phase, use default large radius
    if (currentPhaseIndex >= suspectCameras.length - 1) {
        return 300; // Fallback for last phase
    }
    
    const currentCamera = getCameraById(suspectCameras[currentPhaseIndex].id);
    const nextCamera = getCameraById(suspectCameras[currentPhaseIndex + 1].id);
    
    // Calculate distance between current and next suspect
    const distToNext = distance(currentCamera.x, currentCamera.y, nextCamera.x, nextCamera.y);
    
    // Add 30 pixel buffer so radar extends slightly past the next camera
    const radarRadius = distToNext + 30;
    
    return radarRadius;
}

/**
 * Format seconds to "X.Xs" string
 */
function formatTime(seconds) {
    return seconds.toFixed(1) + 's';
}

/* ============================================================================
   PHASE CALCULATION
   ============================================================================ */

/**
 * Calculate animation phases based on config timing
 * Each phase represents the search from one camera to the next
 */
function computePhases() {
    const suspectCameras = getSuspectCameras();
    const numPhases = suspectCameras.length;
    const phases = [];

    let currentTime = 0;

    for (let i = 0; i < numPhases; i++) {
        let phaseDuration;

        if (i === 0) {
            // First phase: longer for radar to expand slowly
            // Duration matches the first phase radar growth speed
            phaseDuration = 3.5; // Slow expansion for first phase
        } else {
            // Later phases: shorter duration for faster radar expansion
            phaseDuration = 2.5; // Faster expansion for subsequent phases
        }

        phases.push({
            phaseIndex: i,
            cameraId: suspectCameras[i].id,
            startTime: currentTime,
            endTime: currentTime + phaseDuration,
            duration: phaseDuration,
            isFirstPhase: i === 0,
            radarStartTime: currentTime,
            radarEndTime: currentTime + CONFIG.radarGrowthDurationSeconds,
            detectionTime: currentTime + phaseDuration - CONFIG.detectionPauseDurationSeconds,
        });

        currentTime += phaseDuration;
    }

    if (CONFIG.debugLogging) {
        console.log('Animation Phases:', phases);
        phases.forEach(p => {
            console.log(
                `Phase ${p.phaseIndex + 1}: Camera ${p.cameraId} (${formatTime(p.startTime)} → ${formatTime(p.endTime)})`
            );
        });
    }

    return phases;
}

/* ============================================================================
   DOM INITIALIZATION
   ============================================================================ */

/**
 * Initialize all DOM elements: cameras, SVG overlays, etc.
 */
function initializeDOM() {
    // Create camera pins
    const camerasContainer = document.getElementById('cameras-container');
    camerasContainer.innerHTML = '';

    CONFIG.cameras.forEach(camera => {
        const pin = document.createElement('div');
        pin.className = 'camera';
        pin.id = `camera-${camera.id}`;
        pin.setAttribute('data-label', camera.label);
        pin.setAttribute('data-camera-id', camera.id);

        const size = 40 * CONFIG.cameraScale;
        pin.style.width = size + 'px';
        pin.style.height = size + 'px';
        pin.style.left = camera.x - size / 2 + 'px';
        pin.style.top = camera.y - size / 2 + 'px';

        camerasContainer.appendChild(pin);
    });

    // Initialize SVG overlay for radar rings and route line
    const svg = document.getElementById('overlay-svg');
    svg.setAttribute('width', CONFIG.mapWidth);
    svg.setAttribute('height', CONFIG.mapHeight);
    svg.innerHTML = ''; // Clear any previous content

    // Create a defs section for gradients if needed
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.appendChild(defs);
}

/**
 * Create a radar ring (circle) at a given center and radius
 */
function createRadarRing(svg, centerX, centerY, radius) {
    // Create a group to hold all radar layers
    const radarGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    radarGroup.setAttribute('class', 'radar-ring-group');
    
    // Outer glow layer (expanding, fading circle)
    const glow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    glow.setAttribute('cx', centerX);
    glow.setAttribute('cy', centerY);
    glow.setAttribute('r', radius);
    glow.setAttribute('class', 'radar-glow');
    radarGroup.appendChild(glow);
    
    // Main radar ring (solid stroke)
    const mainRing = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    mainRing.setAttribute('cx', centerX);
    mainRing.setAttribute('cy', centerY);
    mainRing.setAttribute('r', radius);
    mainRing.setAttribute('class', 'radar-ring active');
    mainRing.setAttribute('stroke', CONFIG.radarColor);
    mainRing.setAttribute('stroke-width', CONFIG.radarStrokeWidth);
    mainRing.setAttribute('opacity', CONFIG.radarOpacity);
    mainRing.setAttribute('fill', 'none');
    radarGroup.appendChild(mainRing);
    
    // Inner core (bright center point with pulsing animation)
    const core = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    core.setAttribute('cx', centerX);
    core.setAttribute('cy', centerY);
    core.setAttribute('r', 3);
    core.setAttribute('class', 'radar-core');
    radarGroup.appendChild(core);
    
    return radarGroup;
}

/**
 * Draw the route line connecting all suspect cameras in order
 */
function drawRouteLine(svg) {
    const suspectCameras = getSuspectCameras();

    if (suspectCameras.length < 2) return;

    const points = suspectCameras.map(cam => `${cam.x},${cam.y}`).join(' ');

    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', points);
    polyline.setAttribute('class', 'route-line visible');
    polyline.setAttribute('stroke', CONFIG.routeLineColor);
    polyline.setAttribute('stroke-width', CONFIG.routeLineWidth);
    polyline.setAttribute('fill', 'none');

    svg.appendChild(polyline);

    // Add waypoint circles at each suspect camera
    suspectCameras.forEach((cam, index) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', cam.x);
        circle.setAttribute('cy', cam.y);
        circle.setAttribute('r', CONFIG.routeWaypointRadius);
        circle.setAttribute('class', 'route-waypoint visible');
        circle.setAttribute('fill', CONFIG.routeLineColor);
        circle.setAttribute('stroke', '#f1f5f9');
        circle.setAttribute('stroke-width', 2);

        // Stagger the appearance of waypoints
        setTimeout(() => {
            svg.appendChild(circle);
        }, index * 150);
    });
}

/* ============================================================================
   TEXT LABELS
   ============================================================================ */

/**
 * Show a temporary label that fades in and out
 */
function showLabel(text, x, y, duration = 2, cssClass = '') {
    if (!CONFIG.showTextLabels) return;

    const labelsContainer = document.getElementById('labels-container');
    const label = document.createElement('div');
    label.className = 'label ' + cssClass;
    label.textContent = text;
    label.style.left = x + 'px';
    label.style.top = y + 'px';

    labelsContainer.appendChild(label);

    // Trigger animation
    setTimeout(() => label.classList.add('visible'), 10);

    // Remove after duration
    setTimeout(() => {
        label.remove();
    }, duration * 1000 + 1000);
}

/**
 * Show a persistent label that stays visible
 */
function showPersistentLabel(text, x, y, cssClass = '') {
    if (!CONFIG.showTextLabels) return;

    const labelsContainer = document.getElementById('labels-container');
    const label = document.createElement('div');
    label.className = 'label persistent ' + cssClass;
    label.textContent = text;
    label.style.left = x + 'px';
    label.style.top = y + 'px';

    labelsContainer.appendChild(label);
}

/**
 * Clear all labels
 */
function clearLabels() {
    const labelsContainer = document.getElementById('labels-container');
    labelsContainer.innerHTML = '';
}

/* ============================================================================
   ANIMATION UPDATE LOGIC
   ============================================================================ */

/**
 * Update the animation state for the current time
 * This is called on every frame
 */
function updateAnimation(elapsedTime) {
    animationState.currentTime = elapsedTime;

    // Find the current phase
    const currentPhase = animationState.phases.find(
        p => elapsedTime >= p.startTime && elapsedTime < p.endTime
    );

    if (!currentPhase) {
        // Animation is complete
        if (elapsedTime >= CONFIG.animationDurationSeconds) {
            handleAnimationComplete();
        }
        return;
    }

    animationState.currentPhaseIndex = currentPhase.phaseIndex;

    // Reset next-suspect detection flag for new phase
    if (currentPhase.phaseIndex > 0) {
        // When entering a new phase (not the first one), reset the flag for that phase's next suspect
        animationState.nextSuspectDetected = false;
    }

    // ===== PHASE LOGIC =====

    // Set radar center to current suspect camera
    const radarCenter = getCameraById(currentPhase.cameraId);
    animationState.radarCenterCameraId = currentPhase.cameraId;

    // Show radar center indicator
    const radarCenterEl = document.getElementById('radar-center');
    radarCenterEl.classList.add('active');
    radarCenterEl.style.left = radarCenter.x - 4 + 'px';
    radarCenterEl.style.top = radarCenter.y - 4 + 'px';

    // Calculate radar max radius based on distance to next suspect camera + 50px buffer
    const dynamicRadarMaxRadius = calculateRadarMaxRadius(currentPhase.phaseIndex);

    // Radar expansion time: first phase is slower (entire phase duration), later phases are faster (shorter)
    const radarExpansionTime = currentPhase.isFirstPhase ? 
        3.5 :  // First phase: 3.5s for slow expansion
        2.5;   // Later phases: 2.5s for faster expansion

    // Calculate current radar expansion (linear interpolation from min to max radius)
    const radarProgress = Math.max(0, Math.min(1, 
        (elapsedTime - currentPhase.radarStartTime) / radarExpansionTime
    ));
    const currentRadarRadius =
        CONFIG.radarMinRadiusPx +
        (dynamicRadarMaxRadius - CONFIG.radarMinRadiusPx) * radarProgress;

    // Update cameras: mark those within radar radius as "searched"
    const camerasInRadius = getCamerasWithinRadius(
        radarCenter.x,
        radarCenter.y,
        currentRadarRadius
    );

    // Clear previous searched state
    animationState.searchedCameras.forEach(cameraId => {
        const el = document.getElementById(`camera-${cameraId}`);
        if (el && !animationState.detectedCameras.has(cameraId)) {
            el.classList.remove('searched');
        }
    });
    animationState.searchedCameras.clear();

    // Mark cameras in radius as searched
    camerasInRadius.forEach(camera => {
        if (!animationState.detectedCameras.has(camera.id)) {
            animationState.searchedCameras.add(camera.id);
            const el = document.getElementById(`camera-${camera.id}`);
            if (el) {
                el.classList.add('searched');
            }
        }
    });

    // Render radar ring with glow effects
    renderRadarRing(radarCenter.x, radarCenter.y, currentRadarRadius);

    // ===== DETECT CURRENT CAMERA AT START OF PHASE =====
    // Detect the current phase's camera near the start of the phase
    const timeSincePhaseStart = elapsedTime - currentPhase.startTime;
    if (timeSincePhaseStart < 0.5 && !animationState.detectedCameras.has(currentPhase.cameraId)) {
        // Detect this camera near the start of its phase
        detectCamera(currentPhase.cameraId, radarCenter, currentPhase.isFirstPhase);
    }

    // ===== GEOSPATIAL DETECTION LOGIC FOR NEXT CAMERA =====
    // Only detect next camera when it actually enters the search radius
    if (currentPhase.phaseIndex < animationState.phases.length - 1) {
        const nextPhase = animationState.phases[currentPhase.phaseIndex + 1];
        const nextSuspectCamera = getCameraById(nextPhase.cameraId);

        // Check if next suspect is within current radar radius
        const distToNextSuspect = distance(
            radarCenter.x, radarCenter.y,
            nextSuspectCamera.x, nextSuspectCamera.y
        );

        if (distToNextSuspect <= currentRadarRadius && !animationState.nextSuspectDetected) {
            // Next suspect has entered search radius - detect it!
            animationState.nextSuspectDetected = true;
            animationState.nextSuspectDetectionTime = elapsedTime;
            
            // Immediately detect (turn green) the next suspect camera
            detectCamera(nextPhase.cameraId, nextSuspectCamera, false);
        }
    }

    // ===== TEXT LABELS (First Phase) =====
    // All narration is handled in video, no text needed
}

/**
 * Mark a camera as detected (turn it green)
 */
function detectCamera(cameraId, cameraObj, isFirstPhase) {
    animationState.detectedCameras.add(cameraId);

    const el = document.getElementById(`camera-${cameraId}`);
    if (el) {
        el.classList.remove('searched');
        el.classList.add('detected');
        el.classList.add('pulse');

        // Remove pulse animation class after it finishes
        setTimeout(() => {
            el.classList.remove('pulse');
        }, 700);
    }
}

/**
 * Render the radar ring by updating or creating an SVG circle
 */
function renderRadarRing(centerX, centerY, radius) {
    const svg = document.getElementById('overlay-svg');

    // Remove old radar ring group
    const oldRadarGroup = svg.querySelector('.radar-ring-group');
    if (oldRadarGroup) {
        oldRadarGroup.remove();
    }

    // Create new multi-layer radar ring at current radius
    const newRadarGroup = createRadarRing(svg, centerX, centerY, radius);
    svg.appendChild(newRadarGroup);
}

/**
 * Handle animation completion
 */
function handleAnimationComplete() {
    const svg = document.getElementById('overlay-svg');

    // Remove radar ring group
    const radarGroup = svg.querySelector('.radar-ring-group');
    if (radarGroup) {
        radarGroup.remove();
    }

    // Remove radar center indicator
    const radarCenter = document.getElementById('radar-center');
    radarCenter.classList.remove('active');

    // ===== CLEAN UP CAMERA STATES =====
    // Remove all orange "searched" cameras
    animationState.searchedCameras.forEach(cameraId => {
        const el = document.getElementById(`camera-${cameraId}`);
        if (el) {
            el.classList.remove('searched');
        }
    });
    animationState.searchedCameras.clear();

    // Keep all detected cameras green (show the path taken)
    // Remove pulse animation from all detected cameras
    animationState.detectedCameras.forEach(cameraId => {
        const el = document.getElementById(`camera-${cameraId}`);
        if (el) {
            el.classList.remove('pulse');
            el.classList.add('detected'); // Ensure they stay green
        }
    });

    // Draw final route line (no text, video will have narration)
    if (CONFIG.showRouteLine) {
        drawRouteLine(svg);
    }

    // Stop animation
    animationState.isPlaying = false;
    updatePlayButton();
}

/* ============================================================================
   ANIMATION LOOP
   ============================================================================ */

/**
 * Main animation loop using requestAnimationFrame
 */
let animationFrameId = null;

function animationLoop(timestamp) {
    if (!animationState.isPlaying) {
        return;
    }

    if (animationState.startTime === null) {
        animationState.startTime = timestamp;
    }

    const elapsedTime = (timestamp - animationState.startTime) / 1000; // Convert to seconds

    // Update animation
    updateAnimation(elapsedTime);

    // Update UI (progress bar, time display)
    updateUI(elapsedTime);

    // Check if animation is done
    if (elapsedTime < CONFIG.animationDurationSeconds) {
        animationFrameId = requestAnimationFrame(animationLoop);
    } else {
        animationState.isPlaying = false;
        updatePlayButton();
    }
}

/**
 * Update UI elements (progress bar, time display)
 */
function updateUI(elapsedTime) {
    const progressPercent = (elapsedTime / CONFIG.animationDurationSeconds) * 100;
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = Math.min(progressPercent, 100) + '%';

    const currentTimeEl = document.getElementById('current-time');
    currentTimeEl.textContent = Math.min(elapsedTime, CONFIG.animationDurationSeconds).toFixed(1) + 's';

    const totalTimeEl = document.getElementById('total-time');
    totalTimeEl.textContent = CONFIG.animationDurationSeconds.toFixed(1) + 's';
}

/* ============================================================================
   CONTROL FUNCTIONS
   ============================================================================ */

/**
 * Start the animation
 */
function startAnimation() {
    if (animationState.isPlaying) return;

    animationState.isPlaying = true;
    animationState.startTime = null; // Will be set on first frame
    animationFrameId = requestAnimationFrame(animationLoop);
    updatePlayButton();
}

/**
 * Pause the animation
 */
function pauseAnimation() {
    animationState.isPlaying = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    updatePlayButton();
}

/**
 * Reset the animation
 */
function resetAnimation() {
    animationState.isPlaying = false;
    animationState.currentTime = 0;
    animationState.startTime = null;
    animationState.detectedCameras.clear();
    animationState.searchedCameras.clear();
    animationState.radarCenterCameraId = null;
    animationState.currentPhaseIndex = 0;
    animationState.nextSuspectDetected = false;
    animationState.nextSuspectDetectionTime = null;
    animationState.firstPhaseCompleteLabelShown = false;
    animationState.speedConstraintLabelShown = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    // Reset DOM
    clearLabels();
    const svg = document.getElementById('overlay-svg');
    svg.innerHTML = '';
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.appendChild(defs);

    CONFIG.cameras.forEach(camera => {
        const el = document.getElementById(`camera-${camera.id}`);
        if (el) {
            el.classList.remove('searched', 'detected', 'pulse');
        }
    });

    const radarCenter = document.getElementById('radar-center');
    radarCenter.classList.remove('active');

    updateUI(0);
    updatePlayButton();
}

/**
 * Update play button text based on state
 */
function updatePlayButton() {
    const btn = document.getElementById('play-btn');
    if (animationState.isPlaying) {
        btn.textContent = 'Pause Animation';
    } else {
        btn.textContent = 'Play Animation';
    }
}

/* ============================================================================
   EVENT LISTENERS
   ============================================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    initializeDOM();
    animationState.phases = computePhases();
    updateUI(0);
    updatePlayButton();

    // Play button
    const playBtn = document.getElementById('play-btn');
    playBtn.addEventListener('click', () => {
        if (animationState.isPlaying) {
            pauseAnimation();
        } else {
            startAnimation();
        }
    });

    // Reset button
    const resetBtn = document.getElementById('reset-btn');
    resetBtn.addEventListener('click', resetAnimation);

    // Timeline click (seek)
    const timeline = document.querySelector('.timeline');
    timeline.addEventListener('click', (e) => {
        const rect = timeline.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        const seekTime = percent * CONFIG.animationDurationSeconds;

        animationState.currentTime = seekTime;
        animationState.startTime = performance.now() - seekTime * 1000;

        if (!animationState.isPlaying) {
            startAnimation();
        }
    });

    console.log('Animation initialized with config:', CONFIG);
});

/* ============================================================================
   ASSUMPTIONS (Documented in Code)
   ============================================================================ */

/**
 * ASSUMPTION 1: Static Map
 * This animation assumes a static map image; camera positions are in pixels
 * relative to the map container. To change the map, update mapImageUrl in CONFIG.
 *
 * ASSUMPTION 2: Suspect Camera Selection
 * The radar center always moves to the next suspect camera in detectionOrder.
 * Which cameras are suspect locations is controlled by isSuspectLocation and
 * detectionOrder in the cameras array.
 *
 * ASSUMPTION 3: Timing Driven by Config
 * All timings are driven by the CONFIG object. If you change animationDurationSeconds,
 * firstPhaseDurationSeconds, or radarGrowthDurationSeconds, the animation will
 * automatically scale and adjust. The animation is NOT hardcoded to specific times.
 *
 * ASSUMPTION 4: Deterministic Detection
 * The radar search never fails; it always expands until it confirms the suspect at
 * the next camera. This is not a realistic search (with false positives); it's a
 * visualization of the optimal search path.
 */
