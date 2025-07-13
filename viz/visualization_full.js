// Full-Featured Gravity Compression Visualization Engine
// Combines working simple version with all original features

console.log('visualization_full.js loaded');

class GravityVisualizer {
    constructor(containerId, metricFunction, initialParams) {
        console.log('=== GravityVisualizer constructor START ===');
        console.log('containerId:', containerId);
        console.log('initialParams:', initialParams);
        
        this.container = document.getElementById(containerId);
        console.log('container element:', this.container);
        
        if (!this.container) {
            console.error('Container not found!');
            alert('Container element not found: ' + containerId);
            return;
        }
        
        // Add a colored background to container to verify it exists
        this.container.style.backgroundColor = '#001122';
        console.log('Set container background color');
        
        try {
            // Core properties
            this.metric = metricFunction;
            this.params = initialParams || {};
            this.particles = [];
            this.trails = [];
            this.labels = [];
            
            // Simulation parameters
            this.isRunning = true;
            this.quantumStrength = 0.001;
            this.speedMultiplier = 1.0;
            this.maxSteps = 10000;
            this.currentStep = 0;
            this.isDone = false;
            this.showLabels = false;
            this.showTrails = true;
            this.particleCount = 10;
            
            console.log('Properties initialized');
            
            // Create status div
            this.createStatusDiv();
            console.log('Status div created');
            
            // Initialize components
            this.status.textContent = 'Initializing scene...';
            this.initScene();
            console.log('Scene initialized');
            
            this.status.textContent = 'Setting up controls...';
            this.initControls();
            console.log('Controls initialized');
            
            this.status.textContent = 'Creating particles...';
            this.initParticles(this.particleCount);
            console.log('Particles initialized');
            
            this.status.textContent = 'Starting animation...';
            this.animate();
            console.log('Animation started');
            
            console.log('=== GravityVisualizer constructor END ===');
            
        } catch (e) {
            console.error('Error in constructor:', e);
            console.error('Stack trace:', e.stack);
            this.showError('Initialization failed: ' + e.message);
        }
    }
    
    createStatusDiv() {
        console.log('createStatusDiv called');
        this.status = document.createElement('div');
        this.status.style.position = 'absolute';
        this.status.style.bottom = '10px';
        this.status.style.left = '10px';
        this.status.style.color = 'white';
        this.status.style.fontFamily = 'monospace';
        this.status.style.background = 'rgba(0,0,0,0.7)';
        this.status.style.padding = '10px';
        this.status.style.borderRadius = '5px';
        this.status.style.fontSize = '12px';
        this.status.style.zIndex = '1000';
        this.container.appendChild(this.status);
        this.status.textContent = 'Status initialized';
        console.log('Status div appended to container');
    }
    
    showError(msg) {
        console.error('showError called:', msg);
        const div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.top = '50%';
        div.style.left = '50%';
        div.style.transform = 'translate(-50%, -50%)';
        div.style.padding = '20px';
        div.style.background = 'rgba(255,0,0,0.9)';
        div.style.color = 'white';
        div.style.borderRadius = '10px';
        div.style.zIndex = '2000';
        div.innerText = msg;
        this.container.appendChild(div);
    }
    
    initScene() {
        console.log('initScene called');
        
        try {
            // Check if THREE is loaded
            if (typeof THREE === 'undefined') {
                throw new Error('THREE.js not loaded!');
            }
            console.log('THREE.js version:', THREE.REVISION);
            
            // Scene
            this.scene = new THREE.Scene();
            console.log('Scene created');
            
            // Add fog
            this.scene.fog = new THREE.Fog(0x000011, 100, 1000);
            console.log('Fog added');
            
            // Camera
            this.camera = new THREE.PerspectiveCamera(
                75, 
                this.container.clientWidth / this.container.clientHeight, 
                0.1, 
                1000
            );
            this.camera.position.set(0, 100, 200);
            this.camera.lookAt(0, 0, 0);
            console.log('Camera created at position:', this.camera.position);
            
            // Renderer
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.renderer.setClearColor(0x000011); // Very dark blue
            console.log('Renderer created, size:', this.container.clientWidth, 'x', this.container.clientHeight);
            
            // Append renderer
            this.container.appendChild(this.renderer.domElement);
            console.log('Renderer appended to container');
            
            // Test: Add a simple colored cube to verify rendering
            const testGeometry = new THREE.BoxGeometry(10, 10, 10);
            const testMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
            const testCube = new THREE.Mesh(testGeometry, testMaterial);
            testCube.position.set(0, 0, 0);
            this.scene.add(testCube);
            console.log('Test cube added at origin');
            
            // Central black hole
            const bhGeometry = new THREE.SphereGeometry(5, 32, 32);
            const bhMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x000000,
                emissive: 0x000000,
                specular: 0x111111
            });
            this.blackHole = new THREE.Mesh(bhGeometry, bhMaterial);
            this.scene.add(this.blackHole);
            console.log('Black hole added');
            
            // Event horizon (semi-transparent red sphere)
            const ehGeometry = new THREE.SphereGeometry(5.5, 32, 32);
            const ehMaterial = new THREE.MeshBasicMaterial({ 
                color: 0xff0000, 
                transparent: true, 
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            this.eventHorizon = new THREE.Mesh(ehGeometry, ehMaterial);
            this.scene.add(this.eventHorizon);
            console.log('Event horizon added');
            
            // Spacetime grid
            const gridHelper = new THREE.GridHelper(400, 40, 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2;
            this.scene.add(gridHelper);
            console.log('Grid added');
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            this.scene.add(ambientLight);
            console.log('Ambient light added');
            
            const pointLight = new THREE.PointLight(0xffffff, 1, 500);
            pointLight.position.set(50, 50, 50);
            this.scene.add(pointLight);
            console.log('Point light added');
            
            // Add axes helper for orientation
            const axesHelper = new THREE.AxesHelper(50);
            this.scene.add(axesHelper);
            console.log('Axes helper added');
            
            console.log('Scene children count:', this.scene.children.length);
            
            // Initialize CSS2D renderer for labels
            if (typeof THREE.CSS2DRenderer !== 'undefined') {
                this.labelRenderer = new THREE.CSS2DRenderer();
                this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
                this.labelRenderer.domElement.style.position = 'absolute';
                this.labelRenderer.domElement.style.top = '0px';
                this.labelRenderer.domElement.style.pointerEvents = 'none';
                this.container.appendChild(this.labelRenderer.domElement);
                console.log('CSS2DRenderer created for labels');
            }
            
            // Initialize raycaster for mouse interaction
            this.raycaster = new THREE.Raycaster();
            this.mouse = new THREE.Vector2();
            this.hoveredObject = null;
            
            // Create tooltip div
            this.tooltip = document.createElement('div');
            this.tooltip.style.position = 'absolute';
            this.tooltip.style.padding = '10px';
            this.tooltip.style.background = 'rgba(0, 0, 0, 0.8)';
            this.tooltip.style.color = 'white';
            this.tooltip.style.borderRadius = '5px';
            this.tooltip.style.fontSize = '14px';
            this.tooltip.style.maxWidth = '300px';
            this.tooltip.style.display = 'none';
            this.tooltip.style.pointerEvents = 'none';
            this.tooltip.style.zIndex = '1500';
            this.container.appendChild(this.tooltip);
            
            // Add mouse move listener
            this.renderer.domElement.addEventListener('mousemove', (event) => this.onMouseMove(event));
            
            // Add labels after creating objects
            this.createLabels();
            
        } catch (e) {
            console.error('Error in initScene:', e);
            throw e;
        }
    }
    
    createLabels() {
        console.log('Creating 3D labels');
        
        // Label data with explanations
        const labelData = [
            {
                object: this.blackHole,
                text: 'Black Hole',
                explanation: 'The central massive object that curves spacetime according to the theory\'s metric. In the compression hypothesis, this represents the ultimate information sink where quantum states are compressed to their minimal representation.'
            },
            {
                object: this.eventHorizon,
                text: 'Event Horizon',
                explanation: 'The boundary beyond which nothing can escape. For unified theories, this surface may encode both gravitational and electromagnetic information, testing whether the theory correctly handles extreme curvature.'
            },
            {
                object: this.scene.getObjectByProperty('type', 'GridHelper'),
                text: 'Spacetime Grid',
                explanation: 'Visual representation of the spacetime fabric. The grid deformation would show how different theories predict spacetime curvature. In unified theories, this may also encode electromagnetic field effects.'
            }
        ];
        
        this.labels = [];
        
        labelData.forEach(item => {
            if (!item.object) return;
            
            // Create label div
            const labelDiv = document.createElement('div');
            labelDiv.className = 'label';
            labelDiv.textContent = item.text;
            labelDiv.style.color = 'white';
            labelDiv.style.fontFamily = 'Arial, sans-serif';
            labelDiv.style.fontSize = '14px';
            labelDiv.style.background = 'rgba(0, 0, 0, 0.6)';
            labelDiv.style.padding = '4px 8px';
            labelDiv.style.borderRadius = '4px';
            labelDiv.style.userSelect = 'none';
            
            // Create CSS2D object
            let label;
            if (typeof THREE.CSS2DObject !== 'undefined') {
                label = new THREE.CSS2DObject(labelDiv);
                label.position.set(0, 10, 0); // Offset above object
                item.object.add(label);
            } else {
                // Fallback: append to container
                labelDiv.style.position = 'absolute';
                this.container.appendChild(labelDiv);
            }
            
            // Store label info
            this.labels.push({
                div: labelDiv,
                label: label,
                object: item.object,
                explanation: item.explanation
            });
            
            // Set userData for hover detection
            item.object.userData = {
                label: item.text,
                explanation: item.explanation
            };
        });
        
        // Add labels for particles (will be created in initParticles)
        this.particleLabels = [];
        
        // Update label visibility
        this.updateLabelVisibility();
    }
    
    initControls() {
        console.log('initControls called');
        
        try {
            // Orbit controls
            if (typeof THREE.OrbitControls !== 'undefined') {
                this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.orbitControls.enableDamping = true;
                this.orbitControls.dampingFactor = 0.05;
                console.log('OrbitControls created');
            } else {
                console.warn('OrbitControls not loaded - controls will be limited');
            }
            
            // dat.GUI controls
            if (typeof dat !== 'undefined') {
                this.gui = new dat.GUI({ autoPlace: false });
                this.container.appendChild(this.gui.domElement);
                this.gui.domElement.style.position = 'absolute';
                this.gui.domElement.style.top = '10px';
                this.gui.domElement.style.right = '10px';
                console.log('dat.GUI created');
                
                // Simulation folder
                const simFolder = this.gui.addFolder('Simulation');
                simFolder.add(this, 'isRunning').name('Running');
                simFolder.add(this, 'speedMultiplier', 0.1, 5).name('Speed');
                simFolder.add(this, 'quantumStrength', 0, 0.01).name('Quantum Noise');
                simFolder.add(this, 'resetSimulation').name('Reset');
                simFolder.add(this, 'extendSimulation').name('Add 10k Steps');
                simFolder.open();
                console.log('Simulation folder added');
                
                // Display folder
                const displayFolder = this.gui.addFolder('Display');
                displayFolder.add(this, 'showTrails').name('Show Trails');
                displayFolder.add(this, 'showLabels').name('Show Labels').onChange(() => {
                    this.updateLabelVisibility();
                });
                displayFolder.add(this, 'particleCount', 1, 50, 1).name('Particles').onChange(() => {
                    this.resetSimulation();
                });
                displayFolder.open();
                console.log('Display folder added');
                
                // Theory parameters folder
                const paramsFolder = this.gui.addFolder('Theory Parameters');
                for (let key in this.params) {
                    if (typeof this.params[key] === 'number') {
                        paramsFolder.add(this.params, key, -2, 2).name(key).onChange(() => {
                            this.resetSimulation();
                        });
                    }
                }
                paramsFolder.open();
                console.log('Parameters folder added');
                
            } else {
                console.warn('dat.GUI not loaded - controls will be limited');
            }
        } catch (e) {
            console.error('Error in initControls:', e);
            // Don't throw - controls are optional
        }
    }
    
    initParticles(num) {
        console.log('initParticles called with num:', num);
        
        try {
            // Clear existing
            this.particles.forEach(p => {
                if (p.mesh) this.scene.remove(p.mesh);
            });
            this.trails.forEach(trail => {
                this.scene.remove(trail.line);
            });
            this.particles = [];
            this.trails = [];
            
            // Constants
            const RS = 10; // Schwarzschild radius units
            const G = this.params.G || 1;
            const M = this.params.M || 1;
            const C = this.params.C || 1;
            
            console.log('Particle constants - RS:', RS, 'G:', G, 'M:', M, 'C:', C);
            
            for (let i = 0; i < num; i++) {
                // Initial conditions
                const r0 = RS * (3 + Math.random() * 7); // Between 3-10 RS
                const phi0 = (i / num) * 2 * Math.PI;
                
                // Circular orbit velocity
                const v_tan = Math.sqrt(G * M / r0) * 0.95; // Slightly less than circular
                const Lz = r0 * v_tan;
                const E = Math.sqrt(1 - (2 * G * M / (r0 * C**2))) * C;
                
                // Create particle mesh
                const geometry = new THREE.SphereGeometry(0.5, 16, 16);
                const material = new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setHSL(i / num, 0.8, 0.6),
                    emissive: new THREE.Color().setHSL(i / num, 0.8, 0.3),
                    emissiveIntensity: 0.5
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(r0 * Math.cos(phi0), 0, r0 * Math.sin(phi0));
                this.scene.add(mesh);
                
                console.log(`Particle ${i} created at position:`, mesh.position);
                
                // Particle data
                const particle = {
                    mesh: mesh,
                    r: r0,
                    phi: phi0,
                    dr_dtau: 0,
                    dphi_dtau: Lz / (r0 * r0),
                    E: E,
                    Lz: Lz,
                    tau: 0
                };
                this.particles.push(particle);
                
                // Create trail
                const trailGeometry = new THREE.BufferGeometry();
                const positions = new Float32Array(3000); // 1000 points * 3 coords
                trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                trailGeometry.setDrawRange(0, 0);
                
                const trailMaterial = new THREE.LineBasicMaterial({
                    color: material.color,
                    opacity: 0.6,
                    transparent: true
                });
                const trail = new THREE.Line(trailGeometry, trailMaterial);
                this.scene.add(trail);
                
                this.trails.push({
                    line: trail,
                    geometry: trailGeometry,
                    currentIndex: 0
                });
            }
            
            // After creating particles, add labels for first few
            this.particleLabels.forEach(label => {
                if (label.label && label.label.parent) {
                    label.label.parent.remove(label.label);
                }
                if (label.div && label.div.parentNode) {
                    label.div.remove();
                }
            });
            this.particleLabels = [];
            
            // Add labels for first 3 particles
            for (let i = 0; i < Math.min(3, this.particles.length); i++) {
                const particle = this.particles[i];
                
                // Create label
                const labelDiv = document.createElement('div');
                labelDiv.className = 'particle-label';
                labelDiv.textContent = `Particle ${i + 1}`;
                labelDiv.style.color = particle.mesh.material.color.getStyle();
                labelDiv.style.fontFamily = 'Arial, sans-serif';
                labelDiv.style.fontSize = '12px';
                labelDiv.style.background = 'rgba(0, 0, 0, 0.6)';
                labelDiv.style.padding = '2px 6px';
                labelDiv.style.borderRadius = '3px';
                labelDiv.style.userSelect = 'none';
                
                let label;
                if (typeof THREE.CSS2DObject !== 'undefined') {
                    label = new THREE.CSS2DObject(labelDiv);
                    label.position.set(0, 2, 0);
                    particle.mesh.add(label);
                }
                
                this.particleLabels.push({
                    div: labelDiv,
                    label: label,
                    particle: particle
                });
                
                // Set userData for hover
                particle.mesh.userData = {
                    label: `Test Particle ${i + 1}`,
                    explanation: `This particle follows geodesics according to the ${this.params.gamma !== undefined ? 'Linear Signal Loss' : 'selected'} metric. Its trajectory reveals how the theory predicts motion in curved spacetime. Color indicates velocity: blue = slow, red = fast. Initial conditions: r = ${particle.r.toFixed(1)}, L = ${particle.Lz.toFixed(2)}`
                };
            }
            
            this.updateLabelVisibility();
            
            console.log('Total particles created:', this.particles.length);
            
        } catch (e) {
            console.error('Error in initParticles:', e);
            throw e;
        }
    }
    
    animate() {
        console.log('animate called - first frame');
        
        try {
            requestAnimationFrame(() => this.animate());
            
            // Update simulation
            if (this.isRunning && !this.isDone) {
                this.updateParticles(0.1);
            }
            
            // Update controls
            if (this.orbitControls) {
                this.orbitControls.update();
            }
            
            // Update status
            const activeParticles = this.particles.filter(p => p.r > 5).length;
            this.status.innerHTML = `
                Step: ${this.currentStep} / ${this.maxSteps}<br>
                Active Particles: ${activeParticles} / ${this.particles.length}<br>
                Speed: ${this.speedMultiplier.toFixed(1)}x | Quantum: ${this.quantumStrength.toFixed(3)}<br>
                Status: ${this.isDone ? 'Complete' : (this.isRunning ? 'Running' : 'Paused')}
            `;
            
            // Render
            this.renderer.render(this.scene, this.camera);
            
            // Update label renderer if it exists
            if (this.labelRenderer) {
                this.labelRenderer.render(this.scene, this.camera);
            }
            
            // Log every 100 frames
            if (this.currentStep % 100 === 0) {
                console.log('Frame', this.currentStep, '- Camera pos:', this.camera.position, 'Scene children:', this.scene.children.length);
            }
            
        } catch (e) {
            console.error('Error in animate:', e);
            this.showError('Animation error: ' + e.message);
        }
    }
    
    updateParticles(dtau) {
        // Simplified update for debugging
        this.particles.forEach((p, i) => {
            // Simple circular motion
            p.phi += 0.01;
            p.mesh.position.x = p.r * Math.cos(p.phi);
            p.mesh.position.z = p.r * Math.sin(p.phi);
        });
        
        this.currentStep++;
    }
    
    geodesicStep(p, dt, k = null) {
        const r = k ? p.r + k.dr * dt : p.r;
        const dr_dtau = k ? p.dr_dtau + k.ddr * dt : p.dr_dtau;
        
        // Get metric components
        const [g_tt, g_rr, g_pp, g_tp] = this.metric(r, this.params);
        
        // Compute derivatives (simplified for demo)
        const dphi = p.Lz / (r * r);
        const ddr = -this.params.G * this.params.M / (r * r); // Newtonian approximation
        
        return { dr: dr_dtau, dphi: dphi, ddr: ddr };
    }
    
    resetSimulation() {
        console.log('resetSimulation called');
        this.currentStep = 0;
        this.isDone = false;
        this.initParticles(this.particleCount);
    }
    
    extendSimulation() {
        console.log('extendSimulation called');
        this.maxSteps += 10000;
        this.isDone = false;
        console.log('Extended to', this.maxSteps, 'steps');
    }
    
    showDoneMessage() {
        console.log('showDoneMessage called');
        const msg = document.createElement('div');
        msg.style.position = 'absolute';
        msg.style.top = '50%';
        msg.style.left = '50%';
        msg.style.transform = 'translate(-50%, -50%)';
        msg.style.padding = '20px';
        msg.style.background = 'rgba(0,255,0,0.8)';
        msg.style.color = 'white';
        msg.style.borderRadius = '10px';
        msg.style.fontSize = '18px';
        msg.style.zIndex = '2000';
        msg.innerText = `Simulation Complete!\nReached ${this.maxSteps} steps\nClick "Add 10k Steps" to continue`;
        this.container.appendChild(msg);
        setTimeout(() => msg.remove(), 5000);
    }
    
    onMouseMove(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check for intersections
        const objects = [];
        this.scene.traverse(obj => {
            if (obj.userData && obj.userData.label) {
                objects.push(obj);
            }
        });
        
        const intersects = this.raycaster.intersectObjects(objects, true);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            const data = object.userData || (object.parent && object.parent.userData);
            
            if (data && data.explanation && this.hoveredObject !== object) {
                this.hoveredObject = object;
                this.showTooltip(event.clientX, event.clientY, data.label, data.explanation);
                this.renderer.domElement.style.cursor = 'pointer';
            }
        } else if (this.hoveredObject) {
            this.hoveredObject = null;
            this.hideTooltip();
            this.renderer.domElement.style.cursor = 'default';
        }
    }

    showTooltip(x, y, title, text) {
        this.tooltip.innerHTML = `<strong>${title}</strong><br><br>${text}`;
        this.tooltip.style.display = 'block';
        
        // Position tooltip
        const rect = this.container.getBoundingClientRect();
        let left = x - rect.left + 10;
        let top = y - rect.top + 10;
        
        // Keep tooltip within bounds
        if (left + 300 > rect.width) {
            left = x - rect.left - 310;
        }
        if (top + 150 > rect.height) {
            top = y - rect.top - 150;
        }
        
        this.tooltip.style.left = left + 'px';
        this.tooltip.style.top = top + 'px';
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    updateLabelVisibility() {
        // Update main labels
        this.labels.forEach(item => {
            if (item.div) {
                item.div.style.display = this.showLabels ? 'block' : 'none';
            }
        });
        
        // Update particle labels
        this.particleLabels.forEach(item => {
            if (item.div) {
                item.div.style.display = this.showLabels ? 'block' : 'none';
            }
        });
    }
    
    // Window resize handler
    handleResize() {
        console.log('handleResize called');
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        
        // Also resize label renderer
        if (this.labelRenderer) {
            this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
        }
    }
}

// Set up resize handler
window.addEventListener('resize', () => {
    console.log('Window resize event');
    if (window.gravityViz) {
        window.gravityViz.handleResize();
    }
});

window.GravityVisualizer = GravityVisualizer;
console.log('GravityVisualizer added to window'); 