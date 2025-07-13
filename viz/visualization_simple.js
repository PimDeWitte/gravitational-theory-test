// Simplified Gravity Visualization
class GravityVisualizer {
    constructor(containerId, metricFunction, initialParams) {
        console.log('GravityVisualizer constructor called');
        this.container = document.getElementById(containerId);
        if (!this.container) {
            alert('Container element not found: ' + containerId);
            return;
        }
        
        try {
            this.metric = metricFunction;
            this.params = initialParams || {};
            this.particles = [];
            this.isRunning = true;
            
            // Create status div
            this.status = document.createElement('div');
            this.status.style.position = 'absolute';
            this.status.style.top = '10px';
            this.status.style.left = '10px';
            this.status.style.color = 'white';
            this.status.style.fontFamily = 'monospace';
            this.status.style.background = 'rgba(0,0,0,0.5)';
            this.status.style.padding = '5px';
            this.container.appendChild(this.status);
            
            this.status.textContent = 'Initializing Three.js...';
            this.initScene();
            
            this.status.textContent = 'Creating particles...';
            this.initParticles();
            
            this.status.textContent = 'Starting animation...';
            this.animate();
            
        } catch (e) {
            console.error('Error in constructor:', e);
            this.status.textContent = 'ERROR: ' + e.message;
        }
    }
    
    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.container.clientWidth / this.container.clientHeight, 
            0.1, 
            1000
        );
        this.camera.position.set(0, 50, 100);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0x000022); // Dark blue
        this.container.appendChild(this.renderer.domElement);
        
        // Test sphere
        const testGeometry = new THREE.SphereGeometry(5, 32, 32);
        const testMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        this.testSphere = new THREE.Mesh(testGeometry, testMaterial);
        this.scene.add(this.testSphere);
        
        // Grid
        const grid = new THREE.GridHelper(100, 10, 0x444444, 0x222222);
        this.scene.add(grid);
        
        // Lighting
        const light = new THREE.AmbientLight(0xffffff, 1);
        this.scene.add(light);
    }
    
    initParticles() {
        // Create simple orbiting particles
        for (let i = 0; i < 5; i++) {
            const r = 20 + i * 10;
            const angle = (i / 5) * Math.PI * 2;
            
            const geometry = new THREE.SphereGeometry(1, 16, 16);
            const material = new THREE.MeshBasicMaterial({ 
                color: new THREE.Color().setHSL(i / 5, 1, 0.5) 
            });
            const particle = new THREE.Mesh(geometry, material);
            
            particle.position.x = r * Math.cos(angle);
            particle.position.z = r * Math.sin(angle);
            particle.userData = { r, angle, speed: 0.01 };
            
            this.particles.push(particle);
            this.scene.add(particle);
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update status
        this.status.textContent = `Running | Particles: ${this.particles.length} | Frame: ${Date.now()}`;
        
        // Rotate particles
        this.particles.forEach(p => {
            p.userData.angle += p.userData.speed;
            p.position.x = p.userData.r * Math.cos(p.userData.angle);
            p.position.z = p.userData.r * Math.sin(p.userData.angle);
        });
        
        // Rotate test sphere
        this.testSphere.rotation.y += 0.01;
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
}

window.GravityVisualizer = GravityVisualizer; 