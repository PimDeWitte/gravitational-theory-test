class GravityVisualizerEnhanced {
    constructor(containerId, metricFunction, initialParams = {}) {
        console.log('GravityVisualizerEnhanced: Initializing...');
        this.container = document.getElementById(containerId);
        this.metricFunction = metricFunction;
        this.params = { ...initialParams, G: 1, M: 1, C: 1 };
        
        // Simulation state
        this.isRunning = true;
        this.simulationSpeed = 1.0;
        this.showTrails = true;
        this.showLabels = false;
        this.particleCount = 5;
        this.quantumNoise = 0.0;
        this.maxSteps = 100000;
        this.currentStep = 0;
        this.particles = [];
        this.particleNames = [];
        this.selectedParticle = null;
        this.showMetricPanel = false;
        
        // Constants
        this.RS = 2 * this.params.G * this.params.M / (this.params.C * this.params.C);
        this.PARTICLE_SIZE = 0.08; // Larger for clicking
        
        this.init();
    }
    
    async init() {
        console.log('Loading particle names...');
        try {
            const response = await fetch('data/particle_names.json');
            const data = await response.json();
            this.particleNames = data.particles;
        } catch (error) {
            console.warn('Could not load particle names, using defaults:', error);
            this.particleNames = this.generateDefaultNames();
        }
        
        this.setupScene();
        this.setupControls();
        this.setupMetricPanel();
        this.createObjects();
        this.initParticles();
        this.animate();
    }
    
    generateDefaultNames() {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b'];
        return Array.from({length: 20}, (_, i) => ({
            name: `Particle ${i+1}`,
            color: colors[i % colors.length],
            description: `Test particle ${i+1}`,
            symbol: `P${i+1}`
        }));
    }
    
    setupScene() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000033);
        this.scene.fog = new THREE.Fog(0x000033, 50, 200);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.container.clientWidth / this.container.clientHeight, 
            0.1, 
            1000
        );
        this.camera.position.set(0, 30, 50);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // CSS2D Renderer for labels
        this.labelRenderer = new THREE.CSS2DRenderer();
        this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.container.appendChild(this.labelRenderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 1, 100);
        pointLight.position.set(10, 10, 10);
        this.scene.add(pointLight);
        
        // Raycaster for mouse interactions
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Event listeners
        window.addEventListener('resize', () => this.onWindowResize());
        this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
    }
    
    setupMetricPanel() {
        // Create metric tensor panel
        const panel = document.createElement('div');
        panel.id = 'metric-panel';
        panel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            width: 350px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #444;
            border-radius: 5px;
            padding: 15px;
            color: white;
            font-family: monospace;
            font-size: 12px;
            display: none;
            z-index: 1000;
        `;
        
        panel.innerHTML = `
            <h3 style="margin-top: 0;">Metric Tensor at r = <span id="metric-r">0</span> RS</h3>
            <div style="margin-bottom: 10px;">
                Particle: <span id="metric-particle" style="color: #4ecdc4;">None</span>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td class="metric-component" data-component="g_tt" style="padding: 5px; border: 1px solid #666;">
                        g<sub>tt</sub> = <span id="g_tt">0</span>
                    </td>
                    <td class="metric-component" data-component="g_tp" style="padding: 5px; border: 1px solid #666;">
                        g<sub>tφ</sub> = <span id="g_tp">0</span>
                    </td>
                </tr>
                <tr>
                    <td class="metric-component" data-component="g_tp" style="padding: 5px; border: 1px solid #666;">
                        g<sub>φt</sub> = <span id="g_pt">0</span>
                    </td>
                    <td class="metric-component" data-component="g_pp" style="padding: 5px; border: 1px solid #666;">
                        g<sub>φφ</sub> = <span id="g_pp">0</span>
                    </td>
                </tr>
                <tr>
                    <td colspan="2" style="padding: 5px; border: 1px solid #666; text-align: center;">
                        g<sub>rr</sub> = <span id="g_rr">0</span>
                    </td>
                </tr>
            </table>
            <div id="metric-explanation" style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 3px; display: none;">
                <strong id="component-name"></strong><br>
                <span id="component-desc"></span>
            </div>
            <button onclick="window.gravityViz.toggleMetricPanel()" style="margin-top: 10px;">Close</button>
        `;
        
        document.body.appendChild(panel);
        this.metricPanel = panel;
        
        // Add hover effects for metric components
        const components = panel.querySelectorAll('.metric-component');
        components.forEach(comp => {
            comp.style.cursor = 'help';
            comp.addEventListener('mouseenter', (e) => this.showComponentExplanation(e.target.dataset.component));
            comp.addEventListener('mouseleave', () => this.hideComponentExplanation());
        });
    }
    
    showComponentExplanation(component) {
        const explanations = {
            'g_tt': {
                name: 'Time-Time Component (g_tt)',
                desc: 'Measures gravitational time dilation. Negative values indicate timelike signatures. In Schwarzschild metric: g_tt = -(1 - rs/r). Closer to zero means stronger time dilation near the horizon.'
            },
            'g_rr': {
                name: 'Radial-Radial Component (g_rr)',
                desc: 'Measures radial distance distortion. Positive values indicate spacelike signature. In Schwarzschild: g_rr = 1/(1 - rs/r). Diverges at event horizon, indicating coordinate singularity.'
            },
            'g_pp': {
                name: 'Angular-Angular Component (g_φφ)',
                desc: 'Measures circumferential distance. Always r² in spherical symmetry. Represents the "size" of circular orbits at radius r.'
            },
            'g_tp': {
                name: 'Time-Angular Component (g_tφ)',
                desc: 'Cross-term indicating frame-dragging or electromagnetic effects. Zero in Schwarzschild metric. Non-zero values suggest rotation (Kerr) or charge (Reissner-Nordström with rotation).'
            }
        };
        
        const exp = explanations[component];
        if (exp) {
            document.getElementById('component-name').textContent = exp.name;
            document.getElementById('component-desc').textContent = exp.desc;
            document.getElementById('metric-explanation').style.display = 'block';
        }
    }
    
    hideComponentExplanation() {
        document.getElementById('metric-explanation').style.display = 'none';
    }
    
    toggleMetricPanel() {
        this.showMetricPanel = !this.showMetricPanel;
        this.metricPanel.style.display = this.showMetricPanel ? 'block' : 'none';
    }
    
    onMouseClick(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update the picking ray
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check for intersections with particles
        const particleMeshes = this.particles.map(p => p.mesh);
        const intersects = this.raycaster.intersectObjects(particleMeshes);
        
        if (intersects.length > 0) {
            const clickedMesh = intersects[0].object;
            const particle = this.particles.find(p => p.mesh === clickedMesh);
            if (particle) {
                this.selectParticle(particle);
            }
        }
    }
    
    onMouseMove(event) {
        // Update mouse position for hover effects
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }
    
    selectParticle(particle) {
        // Deselect previous particle
        if (this.selectedParticle) {
            this.selectedParticle.mesh.scale.setScalar(1);
        }
        
        // Select new particle
        this.selectedParticle = particle;
        particle.mesh.scale.setScalar(1.5);
        
        // Show metric panel
        this.showMetricPanel = true;
        this.metricPanel.style.display = 'block';
        
        // Update metric panel with particle info
        document.getElementById('metric-particle').textContent = particle.name;
        document.getElementById('metric-particle').style.color = particle.color;
        
        this.updateMetricDisplay(particle);
    }
    
    updateMetricDisplay(particle) {
        if (!particle) return;
        
        const r = Math.sqrt(particle.position.x**2 + particle.position.y**2 + particle.position.z**2);
        const [g_tt, g_rr, g_pp, g_tp] = this.metricFunction(r, this.params);
        
        document.getElementById('metric-r').textContent = (r / this.RS).toFixed(2);
        document.getElementById('g_tt').textContent = g_tt.toFixed(6);
        document.getElementById('g_rr').textContent = g_rr.toFixed(6);
        document.getElementById('g_pp').textContent = g_pp.toFixed(6);
        document.getElementById('g_tp').textContent = g_tp.toFixed(6);
        document.getElementById('g_pt').textContent = g_tp.toFixed(6); // Symmetric
    }
    
    setupControls() {
        const gui = new dat.GUI();
        
        // Simulation controls
        const simFolder = gui.addFolder('Simulation');
        simFolder.add(this, 'isRunning').name('Running');
        simFolder.add(this, 'simulationSpeed', 0.1, 5).name('Speed');
        simFolder.add(this, 'quantumNoise', 0, 0.1).name('Quantum Noise');
        simFolder.add(this, 'reset').name('Reset');
        simFolder.add(this, 'addSteps').name('Add 10k Steps');
        simFolder.open();
        
        // Display controls
        const displayFolder = gui.addFolder('Display');
        displayFolder.add(this, 'showTrails').name('Show Trails');
        displayFolder.add(this, 'showLabels').name('Show Labels').onChange(() => this.updateLabels());
        displayFolder.add(this, 'particleCount', 1, 20, 1).name('Particles').onChange(() => this.updateParticleCount());
        displayFolder.open();
        
        // Theory parameters
        const theoryFolder = gui.addFolder('Theory Parameters');
        for (const [key, value] of Object.entries(this.params)) {
            if (key !== 'G' && key !== 'M' && key !== 'C') {
                theoryFolder.add(this.params, key, -2, 2, 0.01).name(key);
            }
        }
        theoryFolder.open();
        
        this.gui = gui;
    }
    
    createObjects() {
        // Black hole
        const blackHoleGeometry = new THREE.SphereGeometry(this.RS, 32, 32);
        const blackHoleMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x000000,
            emissive: 0x000000
        });
        this.blackHole = new THREE.Mesh(blackHoleGeometry, blackHoleMaterial);
        this.scene.add(this.blackHole);
        
        // Event horizon
        const horizonGeometry = new THREE.SphereGeometry(this.RS * 1.01, 32, 32);
        const horizonMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xff0000,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        this.eventHorizon = new THREE.Mesh(horizonGeometry, horizonMaterial);
        this.scene.add(this.eventHorizon);
        
        // Spacetime grid - horizontal plane
        const gridHelper = new THREE.GridHelper(100, 50, 0x444444, 0x222222);
        gridHelper.rotation.x = Math.PI / 2;
        this.scene.add(gridHelper);
        
        // Add vertical reference planes for 3D visualization
        const planeGeometry = new THREE.PlaneGeometry(100, 100);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x222244,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide,
            wireframe: true
        });
        
        // X-Y plane (vertical)
        const xyPlane = new THREE.Mesh(planeGeometry, planeMaterial);
        xyPlane.rotation.y = Math.PI / 2;
        this.scene.add(xyPlane);
        
        // Y-Z plane (vertical)
        const yzPlane = new THREE.Mesh(planeGeometry, planeMaterial);
        this.scene.add(yzPlane);
        
        // Add axis helpers
        const axesHelper = new THREE.AxesHelper(50);
        this.scene.add(axesHelper);
        
        // Create labels
        this.createLabel(this.blackHole, 'Black Hole', 'The central massive object warping spacetime');
        this.createLabel(this.eventHorizon, 'Event Horizon', 'Point of no return at r = rs');
        this.createLabel(gridHelper, 'Spacetime Grid', 'Visualization of curved spacetime geometry');
    }
    
    createLabel(object, title, description) {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'label';
        labelDiv.style.cssText = `
            background: rgba(0, 0, 0, 0.8);
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-family: Arial;
            font-size: 12px;
            pointer-events: auto;
            cursor: help;
        `;
        labelDiv.textContent = title;
        labelDiv.title = description;
        
        const label = new THREE.CSS2DObject(labelDiv);
        label.position.set(0, 0, 0);
        label.visible = this.showLabels;
        object.add(label);
        
        if (!object.userData) object.userData = {};
        object.userData.label = label;
    }
    
    initParticles() {
        this.particles = [];
        
        for (let i = 0; i < this.particleCount; i++) {
            const nameData = this.particleNames[i % this.particleNames.length];
            const particle = this.createParticle(i, nameData);
            this.particles.push(particle);
        }
    }
    
    createParticle(index, nameData) {
        // Create particle mesh with larger size for clicking
        const geometry = new THREE.SphereGeometry(this.PARTICLE_SIZE, 16, 16);
        const material = new THREE.MeshPhongMaterial({ 
            color: nameData.color,
            emissive: nameData.color,
            emissiveIntensity: 0.5
        });
        const mesh = new THREE.Mesh(geometry, material);
        this.scene.add(mesh);
        
        // Initial conditions with 3D variation
        const r0 = 10 * this.RS + index * 2 * this.RS;
        const angle = (index / this.particleCount) * 2 * Math.PI;
        
        // Add inclination for 3D orbits
        const inclination = (index % 3) * 0.2 - 0.2; // Varies from -0.2 to 0.2 radians
        const y_offset = r0 * Math.sin(inclination) * 0.3; // Vertical offset
        
        const position = new THREE.Vector3(
            r0 * Math.cos(angle) * Math.cos(inclination),
            y_offset,
            r0 * Math.sin(angle) * Math.cos(inclination)
        );
        mesh.position.copy(position);
        
        // Create trail
        const trailGeometry = new THREE.BufferGeometry();
        const trailPositions = new Float32Array(3000); // 1000 points * 3 coordinates
        trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
        const trailMaterial = new THREE.LineBasicMaterial({ 
            color: nameData.color,
            transparent: true,
            opacity: 0.5
        });
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        this.scene.add(trail);
        
        // Create label for particle
        const labelDiv = document.createElement('div');
        labelDiv.className = 'particle-label';
        labelDiv.style.cssText = `
            background: ${nameData.color}88;
            padding: 3px 8px;
            border-radius: 3px;
            color: white;
            font-family: Arial;
            font-size: 11px;
            font-weight: bold;
            pointer-events: none;
        `;
        labelDiv.textContent = `${nameData.symbol} ${nameData.name}`;
        labelDiv.title = nameData.description;
        
        const label = new THREE.CSS2DObject(labelDiv);
        label.position.set(0, this.PARTICLE_SIZE * 2, 0);
        label.visible = this.showLabels;
        mesh.add(label);
        
        return {
            mesh,
            trail,
            label,
            position: position.clone(),
            velocity: this.calculateInitialVelocity(r0, position, index),
            trailIndex: 0,
            trailPositions: [],
            name: nameData.name,
            color: nameData.color,
            description: nameData.description,
            symbol: nameData.symbol,
            active: true,
            angularMomentum: null // Will be calculated from initial conditions
        };
    }
    
    calculateInitialVelocity(r, position, index) {
        // Calculate velocity for inclined orbit
        const v_circular = Math.sqrt(this.params.G * this.params.M / r);
        
        // Direction perpendicular to radius vector
        const r_hat = position.clone().normalize();
        
        // Create a velocity perpendicular to position
        // Use different strategies for different particles to create variety
        let velocity;
        
        if (index % 4 === 0) {
            // Pure circular in x-z plane
            velocity = new THREE.Vector3(-r_hat.z, 0, r_hat.x).multiplyScalar(v_circular * 0.95);
        } else if (index % 4 === 1) {
            // Inclined prograde orbit
            const v_dir = new THREE.Vector3(-r_hat.z, 0.3, r_hat.x).normalize();
            velocity = v_dir.multiplyScalar(v_circular * 0.9);
        } else if (index % 4 === 2) {
            // Retrograde with inclination
            const v_dir = new THREE.Vector3(r_hat.z, -0.2, -r_hat.x).normalize();
            velocity = v_dir.multiplyScalar(v_circular * 0.85);
        } else {
            // Highly inclined elliptical
            const v_dir = new THREE.Vector3(-r_hat.z * 0.7, 0.5, r_hat.x * 0.7).normalize();
            velocity = v_dir.multiplyScalar(v_circular * 0.8);
        }
        
        // Add small random perturbation for variety
        velocity.x += (Math.random() - 0.5) * v_circular * 0.05;
        velocity.y += (Math.random() - 0.5) * v_circular * 0.05;
        velocity.z += (Math.random() - 0.5) * v_circular * 0.05;
        
        return velocity;
    }
    
    updateParticleCount() {
        // Remove extra particles
        while (this.particles.length > this.particleCount) {
            const particle = this.particles.pop();
            this.scene.remove(particle.mesh);
            this.scene.remove(particle.trail);
        }
        
        // Add new particles
        while (this.particles.length < this.particleCount) {
            const index = this.particles.length;
            const nameData = this.particleNames[index % this.particleNames.length];
            const particle = this.createParticle(index, nameData);
            this.particles.push(particle);
        }
    }
    
    updateLabels() {
        // Update main object labels
        [this.blackHole, this.eventHorizon].forEach(obj => {
            if (obj.userData && obj.userData.label) {
                obj.userData.label.visible = this.showLabels;
            }
        });
        
        // Update particle labels
        this.particles.forEach(particle => {
            if (particle.label) {
                particle.label.visible = this.showLabels;
            }
        });
    }
    
    step() {
        if (!this.isRunning || this.currentStep >= this.maxSteps) return;
        
        const dt = 0.001 * this.simulationSpeed;
        
        this.particles.forEach(particle => {
            if (!particle.active) return;
            
            const r = particle.position.length();
            if (r <= this.RS * 1.01) {
                particle.active = false;
                particle.mesh.visible = false;
                return;
            }
            
            // Calculate metric components
            const [g_tt, g_rr, g_pp, g_tp] = this.metricFunction(r, this.params);
            
            // Full 3D relativistic orbital dynamics
            const pos = particle.position;
            const vel = particle.velocity;
            
            // Calculate angular momentum if not set
            if (!particle.angularMomentum) {
                particle.angularMomentum = pos.clone().cross(vel);
            }
            
            // Effective potential approach for relativistic orbits
            const r_hat = pos.clone().normalize();
            const L = particle.angularMomentum.length();
            const L_hat = particle.angularMomentum.clone().normalize();
            
            // Radial acceleration from effective potential
            const F_gravity = -this.params.G * this.params.M / (r * r);
            const F_centrifugal = L * L / (this.params.M * r * r * r);
            
            // Relativistic corrections
            const relativistic_factor = Math.sqrt(Math.abs(g_tt));
            const F_relativistic = 3 * this.params.G * this.params.M * L * L / (this.params.C * this.params.C * r * r * r * r);
            
            // Total radial force
            const F_r = (F_gravity + F_centrifugal - F_relativistic) * relativistic_factor;
            
            // Frame dragging effect from g_tp
            let omega_fd = 0;
            if (g_tp !== 0) {
                omega_fd = -g_tp * this.params.C / (r * r);
            }
            
            // Calculate accelerations
            const a_radial = r_hat.clone().multiplyScalar(F_r);
            
            // Precession of angular momentum due to relativistic effects
            const omega_precession = -3 * this.params.G * this.params.M / (this.params.C * this.params.C * r * r * r);
            
            // Update angular momentum direction (Lense-Thirring precession)
            if (omega_fd !== 0) {
                const dL = L_hat.clone().cross(r_hat).multiplyScalar(omega_fd * L * dt);
                particle.angularMomentum.add(dL);
            }
            
            // Add quantum noise if enabled
            if (this.quantumNoise > 0) {
                a_radial.x += (Math.random() - 0.5) * this.quantumNoise;
                a_radial.y += (Math.random() - 0.5) * this.quantumNoise;
                a_radial.z += (Math.random() - 0.5) * this.quantumNoise;
                
                // Also add noise to angular momentum
                const L_noise = new THREE.Vector3(
                    (Math.random() - 0.5) * this.quantumNoise * L * 0.01,
                    (Math.random() - 0.5) * this.quantumNoise * L * 0.01,
                    (Math.random() - 0.5) * this.quantumNoise * L * 0.01
                );
                particle.angularMomentum.add(L_noise);
            }
            
            // Update velocity
            vel.add(a_radial.multiplyScalar(dt));
            
            // Ensure velocity remains perpendicular to radial direction (conserve angular momentum magnitude)
            const v_radial = r_hat.clone().multiplyScalar(vel.dot(r_hat));
            const v_tangential = vel.clone().sub(v_radial);
            
            // Adjust tangential velocity to conserve angular momentum
            const L_current = pos.clone().cross(v_tangential).length();
            if (L_current > 0) {
                v_tangential.multiplyScalar(L / L_current);
            }
            
            // Reconstruct velocity
            vel.copy(v_radial.add(v_tangential));
            
            // Update position
            pos.add(vel.clone().multiplyScalar(dt));
            
            // Update mesh position
            particle.mesh.position.copy(pos);
            
            // Update trail
            if (this.showTrails) {
                particle.trailPositions.push(pos.clone());
                if (particle.trailPositions.length > 1000) {
                    particle.trailPositions.shift();
                }
                
                const positions = particle.trail.geometry.attributes.position.array;
                for (let i = 0; i < particle.trailPositions.length; i++) {
                    const p = particle.trailPositions[i];
                    positions[i * 3] = p.x;
                    positions[i * 3 + 1] = p.y;
                    positions[i * 3 + 2] = p.z;
                }
                particle.trail.geometry.attributes.position.needsUpdate = true;
                particle.trail.geometry.setDrawRange(0, particle.trailPositions.length);
            }
        });
        
        // Update metric display for selected particle
        if (this.selectedParticle && this.showMetricPanel) {
            this.updateMetricDisplay(this.selectedParticle);
        }
        
        this.currentStep++;
        this.updateStatus();
    }
    
    updateStatus() {
        if (!this.statusDiv) {
            this.statusDiv = document.createElement('div');
            this.statusDiv.style.cssText = `
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: white;
                font-family: monospace;
                font-size: 12px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px;
                border-radius: 5px;
            `;
            this.container.appendChild(this.statusDiv);
        }
        
        const activeCount = this.particles.filter(p => p.active).length;
        this.statusDiv.innerHTML = `
            Step: ${this.currentStep.toLocaleString()} / ${this.maxSteps.toLocaleString()}<br>
            Active Particles: ${activeCount} / ${this.particles.length}<br>
            Speed: ${this.simulationSpeed.toFixed(1)}x<br>
            Quantum Noise: ${this.quantumNoise.toFixed(3)}
        `;
        
        if (this.currentStep >= this.maxSteps) {
            this.statusDiv.innerHTML += '<br><strong>Simulation Complete</strong>';
        }
    }
    
    reset() {
        this.currentStep = 0;
        this.particles.forEach((particle, index) => {
            const r0 = 10 * this.RS + index * 2 * this.RS;
            const angle = (index / this.particleCount) * 2 * Math.PI;
            
            // Recreate 3D initial conditions
            const inclination = (index % 3) * 0.2 - 0.2;
            const y_offset = r0 * Math.sin(inclination) * 0.3;
            
            particle.position.set(
                r0 * Math.cos(angle) * Math.cos(inclination),
                y_offset,
                r0 * Math.sin(angle) * Math.cos(inclination)
            );
            
            particle.velocity = this.calculateInitialVelocity(r0, particle.position, index);
            particle.mesh.position.copy(particle.position);
            particle.active = true;
            particle.mesh.visible = true;
            particle.trailPositions = [];
            particle.trail.geometry.setDrawRange(0, 0);
            particle.angularMomentum = null; // Reset angular momentum
        });
        
        if (this.selectedParticle) {
            this.selectedParticle.mesh.scale.setScalar(1);
            this.selectedParticle = null;
            this.showMetricPanel = false;
            this.metricPanel.style.display = 'none';
        }
    }
    
    addSteps() {
        this.maxSteps += 10000;
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.step();
        this.controls.update();
        
        // Update hover effects
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const particleMeshes = this.particles.map(p => p.mesh);
        const intersects = this.raycaster.intersectObjects(particleMeshes);
        
        // Reset all particles to normal size except selected
        this.particles.forEach(p => {
            if (p !== this.selectedParticle) {
                p.mesh.scale.setScalar(1);
            }
        });
        
        // Highlight hovered particle
        if (intersects.length > 0 && !this.selectedParticle) {
            const hoveredMesh = intersects[0].object;
            hoveredMesh.scale.setScalar(1.2);
            this.renderer.domElement.style.cursor = 'pointer';
        } else {
            this.renderer.domElement.style.cursor = 'default';
        }
        
        this.renderer.render(this.scene, this.camera);
        this.labelRenderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// Export for use
window.GravityVisualizerEnhanced = GravityVisualizerEnhanced; 