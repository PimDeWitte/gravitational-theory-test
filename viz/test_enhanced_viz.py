#!/usr/bin/env python3
"""
Test script for enhanced visualization features.
Demonstrates named particles, metric tensor panel, and interactive features.
"""

import os
import time
import subprocess
import webbrowser

def create_test_visualization():
    """Create a test visualization HTML file with enhanced features."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure data directory exists in parent directory
    os.makedirs(os.path.join(os.path.dirname(script_dir), "data"), exist_ok=True)
    
    # Create test HTML
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced Gravity Visualization Test</title>
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
            max-width: 400px;
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
    </style>
</head>
<body>
    <div id="viz"></div>
    <div id="info-panel">
        <h3>Enhanced Visualization Features</h3>
        <p>This demonstration shows the new interactive features:</p>
        <ul>
            <li><span class="highlight">Named Particles:</span> Each particle has a Greek letter name and description</li>
            <li><span class="highlight">Click Particles:</span> Click any particle to see its metric tensor</li>
            <li><span class="highlight">Metric Panel:</span> Shows real-time g_μν components at particle position</li>
            <li><span class="highlight">Hover Effects:</span> Hover over metric components for explanations</li>
            <li><span class="highlight">Larger Particles:</span> Easier to click and interact with</li>
            <li><span class="highlight">Toggle Labels:</span> Use Display > Show Labels to see particle names</li>
        </ul>
        <p>Try clicking on different particles and watching how the metric tensor changes as they orbit!</p>
    </div>
    <script>
        // Example: Reissner-Nordström metric with charge
        const metricFunction = (r, params) => {
            const rs = 2 * params.G * params.M / params.C ** 2;
            const rq_sq = params.Q * params.Q * params.G / (4 * Math.PI * 8.854e-12 * params.C ** 4);
            const f = 1 - rs / r + rq_sq / (r * r);
            
            // Add small frame-dragging effect for demonstration
            const g_tp = params.rotation * rs / (r * r);
            
            return [-f, 1 / f, r * r, g_tp];
        };
        
        const initialParams = { 
            Q: 1e21,        // Charge parameter
            rotation: 0.1,  // Small rotation to show g_tφ
            G: 1, 
            M: 1, 
            C: 1 
        };
        
        window.gravityViz = new GravityVisualizerEnhanced('viz', metricFunction, initialParams);
        
        // Auto-show labels after a delay
        setTimeout(() => {
            window.gravityViz.showLabels = true;
            window.gravityViz.updateLabels();
            console.log('Labels enabled for demonstration');
        }, 2000);
    </script>
</body>
</html>'''
    
    # Write HTML file
    html_path = os.path.join(script_dir, "test_enhanced_viz.html")
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
        webbrowser.open('http://localhost:8000/viz/test_enhanced_viz.html')
        
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