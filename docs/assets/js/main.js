/**
 * Main JavaScript Controller for CSM-SR Project
 * Handles component loading, initialization, and core functionality
 */

class CSMSRApp {
    constructor() {
        this.components = [
            { id: 'header-component', file: 'components/header.html' },
            { id: 'hero-component', file: 'components/hero.html' },
            { id: 'research-component', file: 'components/research.html' },
            { id: 'results-component', file: 'components/results.html' },
            { id: 'publications-component', file: 'components/publications.html' },
            { id: 'demo-component', file: 'components/demo.html' },
            { id: 'downloads-component', file: 'components/download.html' },
            { id: 'team-component', file: 'components/team.html' },
            { id: 'contact-component', file: 'components/contact.html' },
            { id: 'footer-component', file: 'components/footer.html' }
        ];
        this.loadedComponents = 0;
        this.totalComponents = this.components.length;
    }

    async init() {
        try {
            await this.loadAllComponents();
            await this.loadTeamData();
            this.initializeFeatures();
            this.hideLoadingOverlay();
            console.log('CSM-SR Application initialized successfully');
        } catch (error) {
            console.error('Error initializing application:', error);
            this.showErrorMessage('Failed to load application components');
        }
    }

    async loadAllComponents() {
        const loadPromises = this.components.map(component => 
            this.loadComponent(component.id, component.file)
        );
        
        await Promise.all(loadPromises);
    }

    async loadComponent(elementId, filePath) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`Failed to load ${filePath}: ${response.status}`);
            }
            
            const html = await response.text();
            const element = document.getElementById(elementId);
            
            if (element) {
                element.innerHTML = html;
                this.loadedComponents++;
                this.updateLoadingProgress();
            } else {
                console.warn(`Element with ID '${elementId}' not found`);
            }
        } catch (error) {
            console.error(`Error loading component ${filePath}:`, error);
            // Create fallback content
            const element = document.getElementById(elementId);
            if (element) {
                element.innerHTML = `<div class="alert alert-warning">Component temporarily unavailable</div>`;
            }
        }
    }

    async loadTeamData() {
        try {
            const teamData = await DataLoader.loadTeamData();
            if (teamData && teamData.members) {
                this.renderTeamMembers(teamData.members);
            }
        } catch (error) {
            console.error('Error loading team data:', error);
        }
    }

    renderTeamMembers(members) {
        const teamContainer = document.querySelector('#team .row');
        if (!teamContainer) return;

        const teamHTML = members.map(member => `
            <div class="col-lg-3 col-md-6 mb-4" data-aos="fade-up" data-aos-delay="${Math.random() * 400}">
                <div class="team-card glass-card">
                    <div class="team-image">
                        <img src="${member.photo}" alt="${member.name}" class="img-fluid">
                        <div class="team-overlay">
                            <div class="social-links">
                                ${member.email ? `<a href="mailto:${member.email}" class="social-link"><i class="fas fa-envelope"></i></a>` : ''}
                                ${member.linkedin ? `<a href="${member.linkedin}" target="_blank" class="social-link"><i class="fab fa-linkedin"></i></a>` : ''}
                            </div>
                        </div>
                    </div>
                    <div class="team-info">
                        <h4>${member.name}</h4>
                        <p class="role">${member.role}</p>
                        <p class="bio">${member.bio}</p>
                    </div>
                </div>
            </div>
        `).join('');

        teamContainer.innerHTML = teamHTML;
    }

    initializeFeatures() {
        this.initializeAOS();
        this.initializeNavigation();
        this.initializeScrollEffects();
        this.initializeButtonAnimations();
        this.initializeMetricAnimations();
        this.initializeCopyFeatures();
        this.initializeFormHandling();
    }

    initializeAOS() {
        if (typeof AOS !== 'undefined') {
            AOS.init({
                duration: 1000,
                once: true,
                offset: 100,
                disable: 'mobile'
            });
        }
    }

    initializeNavigation() {
        // Smooth scrolling for navigation links
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href^="#"]');
            if (link) {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });

        // Update active navigation on scroll
        window.addEventListener('scroll', () => {
            this.updateActiveNavigation();
        });
    }

    initializeScrollEffects() {
        // Navbar scroll effect
        window.addEventListener('scroll', () => {
            const navbar = document.getElementById('mainNavbar');
            if (navbar) {
                if (window.scrollY > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
            }
        });

        // Progress bar
        window.addEventListener('scroll', () => {
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
                const scrolled = (window.scrollY / windowHeight) * 100;
                progressBar.style.width = scrolled + '%';
            }
        });
    }

    initializeButtonAnimations() {
        document.addEventListener('click', (e) => {
            const button = e.target.closest('.btn-gradient, .btn-outline-gradient');
            if (button && !button.href?.includes('#')) {
                const originalText = button.innerHTML;
                button.innerHTML = '<span class="loading-animation"></span> Loading...';
                button.disabled = true;
                
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.disabled = false;
                }, 2000);
            }
        });
    }

    initializeMetricAnimations() {
        const animateMetrics = () => {
            const metrics = document.querySelectorAll('.metric-value');
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.animateMetricValue(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            });
            
            metrics.forEach(metric => observer.observe(metric));
        };

        // Initialize after a short delay to ensure components are loaded
        setTimeout(animateMetrics, 500);
    }

    animateMetricValue(target) {
        const finalValue = target.textContent;
        let currentValue = 0;
        const increment = parseFloat(finalValue) / 100;
        
        const updateValue = () => {
            if (currentValue < parseFloat(finalValue)) {
                currentValue += increment;
                if (finalValue.includes('%')) {
                    target.textContent = currentValue.toFixed(1) + '%';
                } else if (finalValue.includes('.')) {
                    target.textContent = currentValue.toFixed(2);
                } else {
                    target.textContent = Math.round(currentValue);
                }
                requestAnimationFrame(updateValue);
            } else {
                target.textContent = finalValue;
            }
        };
        
        updateValue();
    }

    initializeCopyFeatures() {
        document.addEventListener('click', (e) => {
            const codeBlock = e.target.closest('.card-body pre, .citation-block');
            if (codeBlock) {
                const text = codeBlock.textContent;
                this.copyToClipboard(text);
            }
        });
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showNotification('Citation copied to clipboard!', 'success');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            this.showNotification('Failed to copy citation', 'error');
        }
    }

    initializeFormHandling() {
        const contactForm = document.getElementById('contactForm');
        if (contactForm) {
            contactForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFormSubmission(contactForm);
            });
        }
    }

    handleFormSubmission(form) {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        
        // Simulate form submission
        this.showNotification('Message sent successfully!', 'success');
        form.reset();
    }

    updateActiveNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    }

    updateLoadingProgress() {
        const progress = (this.loadedComponents / this.totalComponents) * 100;
        const progressText = document.querySelector('#loadingOverlay p');
        if (progressText) {
            progressText.textContent = `Loading Components... ${Math.round(progress)}%`;
        }
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'success' ? 'success' : 'info'} position-fixed notification-toast`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : 'info'} me-2"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    showErrorMessage(message) {
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            mainContent.innerHTML = `
                <div class="container mt-5 pt-5">
                    <div class="alert alert-danger text-center">
                        <h4><i class="fas fa-exclamation-triangle me-2"></i>Error</h4>
                        <p>${message}</p>
                        <button class="btn btn-primary" onclick="location.reload()">Reload Page</button>
                    </div>
                </div>
            `;
        }
        this.hideLoadingOverlay();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new CSMSRApp();
    app.init();
});

// Handle demo iframe fallback
window.addEventListener('load', () => {
    const iframe = document.querySelector('.demo-embed iframe');
    const fallback = document.querySelector('.iframe-fallback');
    
    if (iframe) {
        iframe.addEventListener('error', () => {
            iframe.style.display = 'none';
            if (fallback) {
                fallback.style.display = 'block';
            }
        });
    }
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CSMSRApp;
}
