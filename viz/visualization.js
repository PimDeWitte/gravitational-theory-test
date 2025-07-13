// Gravity Compression Visualization Engine
// visualization.js - Central file for interactive 3D trajectory visualization
// Uses Three.js for rendering and dat.gui for controls

// Note: This file assumes CDN imports in HTML: Three.js, dat.gui, OrbitControls

class GravityVisualizer {
    constructor(containerId, metricFunction, initialParams) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Container not found');
            return;
        }
        try {
            this.metric = metricFunction; // Function that returns [g_tt, g_rr, g_pp, g_tp] given r, params
            this.params = initialParams || {};
            this.particles = [];
            this.trails = [];
            this.time = 0;
            this.isRunning = true;
            this.quantumStrength = 0.001; // For stochastic quantum effects
            this.speedMultiplier = 1.0; // New: speed control
            this.maxSteps = 10000; // New: max integration steps before 'done'
            this.currentStep = 0;
            this.isDone = false;
            this.showLabels = true; // New: toggle labels
            this.labels = []; // Array for label objects
            this.stepsPerFrame = 10; // Batch integrations
            this.trailUpdateFreq = 5; // Update trails every 5 steps
            this.useWorker = false; // Disable by default for debug
            this.worker = null;
            this.initWorker();
        } catch (e) {
            console.error('Init error:', e);
            this.showError('Initialization failed: ' + e.message);
        }
    }

    showError(msg) {
        const div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.top = '10px';
        div.style.left = '10px';
        div.style.padding = '10px';
        div.style.background = 'rgba(255,0,0,0.8)';
        div.style.color = 'white';
        div.innerText = msg;
        this.container.appendChild(div);
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
        this.camera.position.set(0, 200, 400); // Better initial view
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);
        this.renderer.setClearColor(0x111111); // Dark gray background

        // Central black hole
        const bhGeometry = new THREE.SphereGeometry(5, 32, 32);
        const bhMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        this.blackHole = new THREE.Mesh(bhGeometry, bhMaterial);
        this.scene.add(this.blackHole);

        // Grid helper for spacetime visualization
        const grid = new THREE.GridHelper(200, 20, 0xcccccc, 0xcccccc);
        grid.rotation.x = Math.PI / 2;
        this.scene.add(grid);

        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.8);
        this.scene.add(ambient);
        const pointLight = new THREE.PointLight(0xffffff, 1, 500);
        pointLight.position.set(50, 50, 50);
        this.scene.add(pointLight);

        // Event horizon sphere (semi-transparent)
        const ehGeometry = new THREE.SphereGeometry(5 * 1.01, 32, 32); // Slightly larger than BH
        const ehMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.3 });
        this.eventHorizon = new THREE.Mesh(ehGeometry, ehMaterial);
        this.scene.add(this.eventHorizon);
    }

    initControls() {
        try {
            this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);

            this.gui = new dat.GUI({ autoPlace: false });
            this.container.appendChild(this.gui.domElement);
            this.gui.domElement.style.position = 'absolute';
            this.gui.domElement.style.top = '10px';
            this.gui.domElement.style.right = '10px';

            // Add parameter controls dynamically
            for (let key in this.params) {
                this.gui.add(this.params, key, 0, 2).name(key).onChange(() => this.resetSimulation());
            }
            this.gui.add(this, 'quantumStrength', 0, 0.01).name('Quantum Noise').onChange(() => this.resetSimulation());
            this.gui.add(this, 'isRunning').name('Running');
            this.gui.add(this, 'speedMultiplier', 0.1, 5).name('Sim Speed');
            this.gui.add(this, 'showLabels').name('Show Labels').onChange(() => this.toggleLabels());
            this.gui.add(this, 'resetSimulation').name('Reset');
            this.gui.add(this, 'extendSimulation').name('Continue Sim');
        } catch (e) {
            console.error('Controls error:', e);
        }
    }

    initParticles(num) {
        this.particles = [];
        this.trails.forEach(trail => this.scene.remove(trail));
        this.trails = [];

        const RS = 10; // Schwarzschild radius for scaling
        const G = this.params.G || 1;
        const M = this.params.M || 1;
        const C = this.params.C || 1;
        const rs = 2 * G * M / (C * C);

        for (let i = 0; i < num; i++) {
            const r0 = RS * (3 + Math.random() * 7); // Between 3RS and 10RS
            const phi0 = Math.random() * 2 * Math.PI;
            const theta = Math.PI / 2; // Equatorial

            // Compute initial angular momentum for circular orbit approx
            const v_tan = Math.sqrt(G * M / r0) * 0.8; // Stable orbit velocity
            const Lz = r0 * v_tan;
            const E = Math.sqrt(1 - (2 * G * M / (r0 * C**2)) * (1 + (Lz**2 / (r0**2 * C**2)))) * C; // Approx

            // Position in Cartesian
            const x = r0 * Math.cos(phi0);
            const y = r0 * Math.sin(phi0);
            const z = 0;
            const pos = new THREE.Vector3(x, y, z);

            // Initial velocity (tangential)
            const vx = -v_tan * Math.sin(phi0);
            const vy = v_tan * Math.cos(phi0);
            const vz = 0;
            const vel = new THREE.Vector3(vx, vy, vz);

            // For geodesic: track r, phi, dr/dtau, dphi/dtau
            const particle = {
                r: r0,
                phi: phi0,
                dr_dtau: 0, // radial velocity
                dphi_dtau: Lz / (r0**2), // angular
                E: E,
                Lz: Lz,
                tau: 0,
                pos: pos,
                vel: vel
            };

            this.particles.push(particle);

            // Trail
            const color = new THREE.Color().setHSL(i / num, 0.8, 0.6);
            const trailMaterial = new THREE.LineBasicMaterial({ color: color });
            const trailGeometry = new THREE.BufferGeometry().setFromPoints([pos.clone()]);
            const trail = new THREE.Line(trailGeometry, trailMaterial);
            this.scene.add(trail);
            this.trails.push(trail);
        }
    }

    initLabels() {
        try {
            const loader = new THREE.FontLoader();
            loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
                const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 });

                // Black Hole label
                const bhLabel = this.createTextMesh(font, 'Black Hole\nCentral mass encoding\ncompressed information', textMaterial);
                bhLabel.position.set(0, 0, 10);
                this.scene.add(bhLabel);
                this.labels.push(bhLabel);

                // Grid label
                const gridLabel = this.createTextMesh(font, 'Spacetime Grid\nVisualizes the curved\nfabric of reality', textMaterial);
                gridLabel.position.set(50, 0, -50);
                gridLabel.rotation.set(0, Math.PI / 4, 0);
                this.scene.add(gridLabel);
                this.labels.push(gridLabel);

                // Trail label (example for first trail)
                const trailLabel = this.createTextMesh(font, 'Particle Trail\nShows trajectory under\ntheory\'s metric', textMaterial);
                trailLabel.position.set(30, 0, 30);
                this.scene.add(trailLabel);
                this.labels.push(trailLabel);
            }, undefined, (err) => {
                console.warn('Font load failed:', err);
                this.showLabels = false;
            });
        } catch (e) {
            console.error('Labels init error:', e);
            this.showLabels = false;
        }
    }

    createTextMesh(font, text, material) {
        const shapes = font.generateShapes(text, 5);
        const geometry = new THREE.ShapeBufferGeometry(shapes);
        geometry.computeBoundingBox();
        const xMid = -0.5 * (geometry.boundingBox.max.x - geometry.boundingBox.min.x);
        geometry.translate(xMid, 0, 0);
        return new THREE.Mesh(geometry, material);
    }

    toggleLabels() {
        this.labels.forEach(label => { label.visible = this.showLabels; });
    }

    resetSimulation() {
        this.time = 0;
        this.currentStep = 0;
        this.isDone = false;
        this.initParticles(this.particles.length);
    }

    // RK4 step for geodesic (simplified for equatorial, photon-like or timelike)
    rk4Step(p, dtau) {
        const r = p.r;
        const [g_tt, g_rr, g_pp, g_tp] = this.metric(r, this.params);

        const k1 = this.ode(r, p.dr_dtau, g_tt, g_rr, g_pp, g_tp, p.E, p.Lz);
        const mid_r1 = r + 0.5 * dtau * k1[0];
        const mid_dr1 = p.dr_dtau + 0.5 * dtau * k1[1];
        const k2 = this.ode(mid_r1, mid_dr1, ...this.metric(mid_r1, this.params), p.E, p.Lz);

        const mid_r2 = r + 0.5 * dtau * k2[0];
        const mid_dr2 = p.dr_dtau + 0.5 * dtau * k2[1];
        const k3 = this.ode(mid_r2, mid_dr2, ...this.metric(mid_r2, this.params), p.E, p.Lz);

        const mid_r3 = r + dtau * k3[0];
        const mid_dr3 = p.dr_dtau + dtau * k3[1];
        const k4 = this.ode(mid_r3, mid_dr3, ...this.metric(mid_r3, this.params), p.E, p.Lz);

        const dr = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * (dtau / 6);
        const ddr = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * (dtau / 6);

        p.r += dr;
        p.phi += p.dphi_dtau * dtau;
        p.dr_dtau += ddr;
        p.tau += dtau;

        // Update 3D position
        p.pos.x = p.r * Math.cos(p.phi);
        p.pos.y = p.r * Math.sin(p.phi);
        p.pos.z = 0;
    }

    // In ode, fix dV_dr approximation
    ode(r, dr_dtau, g_tt, g_rr, g_pp, g_tp, E, Lz) {
        const V_sq = this.computeV(r, g_tt, g_rr, g_pp, g_tp, E, Lz);
        const eps = 1e-5;
        const [g_tt_p, g_rr_p, g_pp_p, g_tp_p] = this.metric(r + eps, this.params);
        const V_sq_p = this.computeV(r + eps, g_tt_p, g_rr_p, g_pp_p, g_tp_p, E, Lz);
        const [g_tt_m, g_rr_m, g_pp_m, g_tp_m] = this.metric(r - eps, this.params);
        const V_sq_m = this.computeV(r - eps, g_tt_m, g_rr_m, g_pp_m, g_tp_m, E, Lz);
        const dV_dr = (V_sq_p - V_sq_m) / (2 * eps);
        const d2r_dtau2 = 0.5 * dV_dr;
        return [dr_dtau, d2r_dtau2];
    }

    computeV(r, g_tt, g_rr, g_pp, g_tp, E, Lz) { // Note: Parameters passed, but for finite diff, we need to recompute metric
        // Actually, for accurate finite diff, ode needs access to metric
        // But since g_ components depend on r, we can't use fixed ones
        // Fix: Pass metric to ode, but since it's approx, use central diff with metric calls
        // Wait, in main thread, computeV takes fixed g_, but that's incorrect for diff
        // Proper fix: ode takes metric func
        const det = g_tp**2 - g_tt * g_pp;
        if (Math.abs(det) < 1e-10) return 0;
        const u_t = (E * g_pp + Lz * g_tp) / det;
        const u_phi = -(E * g_tp + Lz * g_tt) / det;
        return (-1 - (g_tt * u_t**2 + g_pp * u_phi**2 + 2 * g_tp * u_t * u_phi)) / g_rr;
    }

    initWorker() {
        if (this.useWorker) {
            this.worker = new Worker(URL.createObjectURL(new Blob([`
                function computeV(r, g_tt, g_rr, g_pp, g_tp, E, Lz) {
                    const det = g_tp**2 - g_tt * g_pp;
                    if (Math.abs(det) < 1e-10) return 0;
                    const u_t = (E * g_pp + Lz * g_tp) / det;
                    const u_phi = -(E * g_tp + Lz * g_tt) / det;
                    return (-1 - (g_tt * u_t**2 + g_pp * u_phi**2 + 2 * g_tp * u_t * u_phi)) / g_rr;
                }

                function ode(r, dr_dtau, metric, params, E, Lz) {
                    const [g_tt, g_rr, g_pp, g_tp] = metric(r, params);
                    const V_sq = computeV(r, g_tt, g_rr, g_pp, g_tp, E, Lz);
                    const eps = 1e-5;
                    const [g_tt_p, g_rr_p, g_pp_p, g_tp_p] = metric(r + eps, params);
                    const V_sq_p = computeV(r + eps, g_tt_p, g_rr_p, g_pp_p, g_tp_p, E, Lz);
                    const [g_tt_m, g_rr_m, g_pp_m, g_tp_m] = metric(r - eps, params);
                    const V_sq_m = computeV(r - eps, g_tt_m, g_rr_m, g_pp_m, g_tp_m, E, Lz);
                    const dV_dr = (V_sq_p - V_sq_m) / (2 * eps);
                    const d2r_dtau2 = 0.5 * dV_dr;
                    return [dr_dtau, d2r_dtau2];
                }

                self.onmessage = function(e) {
                    const { particles, metricCode, params, dtau, steps } = e.data;
                    const metric = new Function('r', 'params', metricCode);
                    particles.forEach(p => {
                        for (let s = 0; s < steps; s++) {
                            const [dr, ddr] = ode(p.r, p.dr_dtau, metric, params, p.E, p.Lz);
                            p.r += dr * dtau;
                            p.phi += p.dphi_dtau * dtau;
                            p.dr_dtau += ddr * dtau;
                            p.tau += dtau;
                            p.pos.x = p.r * Math.cos(p.phi);
                            p.pos.y = p.r * Math.sin(p.phi);
                            p.pos.z = 0;
                        }
                    });
                    self.postMessage(particles);
                };
            `], { type: 'application/javascript' })));
            this.worker.onmessage = (e) => {
                this.particles = e.data;
                this.updateTrails();
                this.updateColors();
            };
        }
    }

    updateParticles(dtau) {
        if (this.isDone) return;

        dtau *= this.speedMultiplier;

        if (this.useWorker) {
            this.worker.postMessage({
                particles: this.particles,
                metricCode: this.metric.toString(), // Send metric function as string
                params: this.params,
                dtau: dtau,
                steps: this.stepsPerFrame
            });
        } else {
            for (let s = 0; s < this.stepsPerFrame; s++) {
                this.particles.forEach((p, i) => {
                    this.rk4Step(p, dtau);
                    if (this.quantumStrength > 0) {
                        p.dr_dtau += (Math.random() - 0.5) * this.quantumStrength * Math.sqrt(p.E);
                    }
                    if (s % this.trailUpdateFreq === 0) {
                        this.updateTrail(i);
                    }
                });
                this.currentStep++;
                if (this.currentStep >= this.maxSteps) {
                    this.isDone = true;
                    this.showDoneMessage();
                    return;
                }
            }
            this.updateColors(); // Update colors after batch
        }
    }

    updateTrail(i) {
        const p = this.particles[i];
        const positions = this.trails[i].geometry.attributes.position.array;
        // Use typed arrays for efficiency
        const newPositions = new Float32Array(positions.length + 3);
        newPositions.set(positions);
        newPositions[newPositions.length - 3] = p.pos.x;
        newPositions[newPositions.length - 2] = p.pos.y;
        newPositions[newPositions.length - 1] = p.pos.z;
        if (newPositions.length > 1000 * 3) {
            newPositions.set(newPositions.subarray(3));
        }
        this.trails[i].geometry.setAttribute('position', new THREE.BufferAttribute(newPositions, 3));
        this.trails[i].geometry.attributes.position.needsUpdate = true;
    }

    updateColors() {
        this.particles.forEach((p, i) => {
            const speed = Math.sqrt(p.dr_dtau**2 + (p.r * p.dphi_dtau)**2);
            const hue = 0.666 - Math.min(speed / 2, 0.666);
            this.trails[i].material.color.setHSL(hue, 1, 0.5);
        });
    }

    showDoneMessage() {
        const msg = document.createElement('div');
        msg.style.position = 'absolute';
        msg.style.top = '50%';
        msg.style.left = '50%';
        msg.style.transform = 'translate(-50%, -50%)';
        msg.style.padding = '20px';
        msg.style.background = 'rgba(0,0,0,0.7)';
        msg.style.color = 'white';
        msg.style.borderRadius = '10px';
        msg.innerText = 'Simulation Complete\n(Reached max steps: ' + this.maxSteps + ')\nToggle Running to continue';
        this.container.appendChild(msg);
        setTimeout(() => msg.remove(), 5000);
    }

    extendSimulation() {
        this.maxSteps += 10000;
        this.isDone = false;
        console.log('Extended max steps to:', this.maxSteps);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        try {
            console.log('Frame:', this.currentStep, 'Particles:', this.particles.length, 'First pos:', this.particles[0]?.pos);
            if (this.isRunning && !this.isDone) {
                this.updateParticles(0.1);
            }
            this.orbitControls?.update();
            this.renderer.render(this.scene, this.camera);
            console.log('Rendered frame', this.currentStep);
        } catch (e) {
            console.error('Animate error:', e);
            this.showError('Render error: ' + e.message);
        }
    }
}

// Export for use in HTML
window.GravityVisualizer = GravityVisualizer; 