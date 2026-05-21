# Geospatially Constrained Search Animation

A single-page HTML/CSS/JS animation that simulates an optimized radar-based search for a tracking project. The animation demonstrates how a system expands its search radius from a starting camera to locate a suspect across multiple cameras in sequence.

## Files

- **index.html** — Main HTML page with container and controls
- **heatmap-anim.css** — Styling for cameras, radar rings, labels, and route lines
- **heatmap-anim.js** — Animation engine with all configuration at the top

## Quick Start

1. Open `index.html` in a modern web browser
2. Click **Play Animation** to start the demo
3. Use **Reset** to restart
4. Click the progress bar to seek to a specific time

## How It Works

### The Scene

- **Static map** with 12–20 camera pins displayed as gray circular markers
- **4 suspect cameras** where the suspect is detected, distributed across the map
- **Radar search** that expands from each suspect camera to find the next one

### Animation Flow

1. **Phase 1 (First Detection)**: The radar starts at the first suspect camera and expands outward, searching nearby cameras until it confirms the suspect.
   - Text label: *"This is the speed constraint of the subject"*
   - When detected: *"This is where the subject started"*
   - This phase is longer (3.5s by default) to help viewers understand the logic

2. **Phases 2–4 (Subsequent Detections)**: The radar moves to the next suspect camera and repeats the search. These phases are faster and more compact.
   - Each detected camera turns green
   - Cameras within the search radius turn light gray

3. **Final Route**: After all 4 detections, a blue route line connects the suspect cameras in order, showing the suspect's path.
   - Text label: *"This is where we found the suspect"*

### Camera States

- **Gray** (inactive): Not yet searched
- **Light Gray** (searched): Currently within the radar search radius
- **Green** (detected): Confirmed suspect location
- **Green + Pulse**: Newly detected (animation pulse for emphasis)

## Customization

All configuration is at the top of `heatmap-anim.js` in the `CONFIG` object. No logic changes needed.

### Common Changes

#### 1. **Change the Number of Cameras**

Edit the `cameras` array in `CONFIG`:

```javascript
cameras: [
    // Suspect cameras
    { id: 1, x: 100, y: 150, label: '1', isSuspectLocation: true, detectionOrder: 1 },
    { id: 5, x: 350, y: 200, label: '5', isSuspectLocation: true, detectionOrder: 2 },
    // Add more cameras here...

    // Non-suspect cameras
    { id: 2, x: 180, y: 100, label: '2', isSuspectLocation: false, detectionOrder: 0 },
    // ...
]
```

- `id`: Unique camera identifier (any integer)
- `x, y`: Pixel position on the map (0,0 is top-left)
- `label`: Number or name displayed inside the pin
- `isSuspectLocation`: `true` for the 4 suspect cameras, `false` for others
- `detectionOrder`: 1–4 for suspect cameras (order in which they are detected); 0 for others

#### 2. **Change Which Cameras Are Suspects**

Set `isSuspectLocation: true` and assign a `detectionOrder` (1–4) to exactly 4 cameras:

```javascript
{ id: 7, x: 500, y: 250, label: '7', isSuspectLocation: true, detectionOrder: 2 },
```

The `detectionOrder` determines the sequence in which the radar finds them.

#### 3. **Change Animation Duration**

```javascript
animationDurationSeconds: 9,          // Total time (change to 5, 8, 10, etc.)
firstPhaseDurationSeconds: 3.5,       // Time for first phase (longer to explain logic)
```

The remaining phases are split evenly. E.g., if total=9 and first=3.5, each of phases 2–4 gets (9-3.5)/3 ≈ 1.83s.

#### 4. **Change Radar Expansion Speed**

```javascript
radarGrowthDurationSeconds: 1.2,    // Time for radar to expand from min to max
radarMinRadiusPx: 20,               // Starting radius (pixels)
radarMaxRadiusPx: 150,              // Ending radius (pixels)
searchSpeedFactor: 1.0,             // Multiplier (1.0=normal, 2.0=twice as fast)
```

#### 5. **Change Colors**

```javascript
radarColor: '#00ff00',              // Radar ring (green)
inactiveCameraColor: '#64748b',     // Inactive camera (gray)
searchingCameraColor: '#94a3b8',    // Searched camera (light gray)
detectedCameraColor: '#10b981',     // Detected camera (green)
routeLineColor: '#0ea5e9',          // Route line (cyan)
```

#### 6. **Change Text Labels**

To disable text labels entirely:

```javascript
showTextLabels: false,
```

To customize label text, edit the `showLabel()` calls in the `updateAnimation()` function in `heatmap-anim.js`.

#### 7. **Change Map Background**

To use a custom map image:

```javascript
mapImageUrl: 'https://your-domain.com/map.png',
// or use a data URL:
// mapImageUrl: 'data:image/png;base64,iVBORw0KG...',
```

To use a simple placeholder (default), leave it empty:

```javascript
mapImageUrl: '',
```

#### 8. **Change Map Dimensions**

```javascript
mapWidth: 800,    // Width in pixels
mapHeight: 600,   // Height in pixels
```

Then update the aspect ratio in `heatmap-anim.css` if needed:

```css
.map-container {
    aspect-ratio: 4 / 3;  /* 800x600 ratio */
}
```

#### 9. **Disable Route Line**

```javascript
showRouteLine: false,
```

#### 10. **Change Camera Pin Size**

```javascript
cameraScale: 1.0,   // 1.0 = default, 1.5 = 50% larger, 0.8 = 20% smaller
```

### Advanced: Debugging

Enable debug logging to see phase timings in the browser console:

```javascript
debugLogging: true,
```

Then open the browser's Developer Tools (F12) to see logs like:

```
Animation Phases: [...]
Phase 1: Camera 1 (0.0s → 3.5s)
Phase 2: Camera 5 (3.5s → 5.3s)
Phase 3: Camera 10 (5.3s → 7.1s)
Phase 4: Camera 14 (7.1s → 9.0s)
```

## Technical Details

### Animation Engine

- **Timing**: Uses `requestAnimationFrame` for smooth 60 FPS animation
- **Radar Expansion**: Linear interpolation from `radarMinRadiusPx` to `radarMaxRadiusPx`
- **Camera Search**: Computed every frame based on distance from radar center
- **SVG Rendering**: Radar rings and route lines are drawn on an SVG canvas overlay

### Data Flow

1. Config is parsed at page load
2. Phases are computed based on timing and suspect camera order
3. Animation loop updates radar radius and camera states each frame
4. Detected cameras are marked green; cameras within radius are marked light gray
5. On completion, the route line is drawn

### Assumptions

1. **Static Map**: Camera positions are fixed pixels relative to the map container
2. **Radar Success**: The search never fails; it always finds the next suspect
3. **Geospatial Constraint**: Only cameras within the growing radar radius are searched
4. **Linear Timing**: All phases follow a consistent linear expansion pattern

## Browser Compatibility

- Chrome/Chromium ≥ 60
- Firefox ≥ 55
- Safari ≥ 12
- Edge ≥ 79

Modern browsers with ES6 support and SVG capabilities are required.

## Troubleshooting

**Animation doesn't start**
- Check browser console for errors (F12)
- Ensure JavaScript is enabled
- Try a different browser

**Cameras not visible**
- Check that `CONFIG.mapWidth` and `CONFIG.mapHeight` match the CSS container size
- Verify camera `x` and `y` values are within the map bounds

**Radar not expanding**
- Check `radarGrowthDurationSeconds` and `radarMinRadiusPx` / `radarMaxRadiusPx`
- Ensure `CONFIG.showTextLabels` is not blocking the radar render

**Route line not showing**
- Set `showRouteLine: true`
- Verify at least 2 cameras have `isSuspectLocation: true`

## License

Free to use and modify. No attribution required.
