#!/usr/bin/env python3
"""
Test script for 3D orbital motion visualization.
Demonstrates particles in 3D orbits with various inclinations using Kerr metric.
"""

import os
import time
import subprocess
import webbrowser

def create_test_visualization():
    """Create a test visualization HTML file with 3D orbital motion."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure data directory exists in parent directory
    os.makedirs(os.path.join(os.path.dirname(script_dir), "data"), exist_ok=True)
    
    # Create test HTML with Kerr metric and 3D particles
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>3D Orbital Motion Test - Kerr Metric</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/renderers/CSS2DRenderer.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <script src="visualization_enhanced.js"></script>
    <style>
        body { 
            margin: 0; 
            font-family: Arial, sans-serif;
        } 
        #viz { 
            width: 100vw; 
            height: 100vh; 
        }
        .label:hover {
            background: rgba(0, 0, 0, 0.95) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        }
        .metric-component:hover {
            background: rgba(100, 100, 255, 0.2);
        }
        #info-panel {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-size: 14px;
            max-width: 450px;
        }
        #info-panel h3 {
            margin-top: 0;
            color: #4ecdc4;
        }
        #info-panel ul {
            padding-left: 20px;
        }
        #info-panel li {
            margin: 5px 0;
        }
        .highlight {
            color: #f9ca24;
            font-weight: bold;
        }
        .code {
            background: rgba(255,255,255,0.1);
            padding: 2px 4px;
            font-family: monospace;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div id="viz"></div>
    <div id="info-panel">
        <h3>3D Orbital Motion Demonstration</h3>
        <p>This visualization shows particles in <span class="highlight">full 3D orbits</span> around a rotating black hole (Kerr metric).</p>
        
        <h4>What's Different:</h4>
        <ul>
            <li><span class="highlight">Inclined Orbits:</span> Particles start at different inclinations relative to the equatorial plane</li>
            <li><span class="highlight">Frame Dragging:</span> The rotating black hole drags spacetime, causing orbital precession</li>
            <li><span class="highlight">3D Trajectories:</span> Particles move in all three dimensions (x, y, z)</li>
            <li><span class="highlight">Angular Momentum:</span> Each particle conserves its angular momentum vector (not just magnitude)</li>
        </ul>
        
        <h4>Visual Aids:</h4>
        <ul>
            <li>Three orthogonal grid planes (XY, YZ, XZ)</li>
            <li>RGB axis indicators (Red=X, Green=Y, Blue=Z)</li>
            <li>Particle trails show full 3D paths</li>
        </ul>
        
        <p>Try rotating the camera (click and drag) to see the 3D structure!</p>
        <p>Metric: <span class="code">g_tφ = a*rs/(r²+a²)</span> (frame dragging)</p>
    </div>
    <script>
        // Kerr metric (rotating black hole) for demonstration
        const metricFunction = (r, params) => {
            const rs = 2 * params.G * params.M / params.C ** 2;
            const a = params.spin * rs; // Spin parameter
            const r2 = r * r;
            const a2 = a * a;
            const Sigma = r2 + a2;
            const Delta = r2 - rs * r + a2;
            
            // Simplified Kerr metric components in Boyer-Lindquist coordinates
            const g_tt = -(1 - rs * r / Sigma);
            const g_rr = Sigma / Delta;
            const g_pp = Sigma; // Simplified for θ=π/2
            const g_tp = -a * rs * r / Sigma; // Frame dragging term
            
            return [g_tt, g_rr, g_pp, g_tp];
        };
        
        const initialParams = { 
            spin: 0.5,      // Dimensionless spin parameter (0 to 1)
            G: 1, 
            M: 1, 
            C: 1 
        };
        
        window.gravityViz = new GravityVisualizerEnhanced('viz', metricFunction, initialParams);
        
        // Start with more particles to show variety
        setTimeout(() => {
            window.gravityViz.particleCount = 8;
            window.gravityViz.updateParticleCount();
            window.gravityViz.showLabels = true;
            window.gravityViz.updateLabels();
            console.log('Increased particle count and enabled labels');
        }, 1000);
        
        // Adjust camera for better 3D view
        setTimeout(() => {
            window.gravityViz.camera.position.set(30, 40, 60);
            window.gravityViz.camera.lookAt(0, 0, 0);
            window.gravityViz.controls.update();
        }, 500);
    </script>
</body>
</html>'''
    
    # Write HTML file
    html_path = os.path.join(script_dir, "test_3d_motion.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"Created test visualization: {html_path}")
    
    # Start local server from parent directory
    print("Starting local server on port 8000...")
    parent_dir = os.path.dirname(script_dir)
    try:
        # Change to parent directory for server
        original_dir = os.getcwd()
        os.chdir(parent_dir)
        
        server = subprocess.Popen(
            ['python3', '-m', 'http.server', '8000'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(1)  # Give server time to start
        
        print("Opening browser...")
        webbrowser.open('http://localhost:8000/viz/test_3d_motion.html')
        
        print("\nServer running. Press Ctrl+C to stop.")
        server.wait()
        
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.terminate()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    create_test_visualization() 