/**
 * Animation and Visual Effects for CSM-SR Project
 * Handles all animations, transitions, and interactive effects
 */

class AnimationController {
    constructor() {
        this.observers = new Map();
        this.animationQueue = [];
        this.isAnimating = false;
        this.floatingElements = [];
        this.particleSystem = null;
    }

    init() {
        this.initializeScrollAnimations();
        this.initializeHoverEffects();
        this.initializeParallaxEffects();
        this.initializeLoadingAnimations();
        this.initializeFloatingElements();
        this.initializeTypewriterEffect();
        this.initializeProgressBars();
        this.initializeCounterAnimations();
        this.initializeParticleEffects();
        console.log('Animation Controller initialized');
    }

    initializeScrollAnimations() {
        // Fade in animation on scroll
        const fadeElements = document.querySelectorAll('[data-animation="fade-in"]');
        this.createScrollObserver(fadeElements, (element) => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'all 0.6s ease-out';
        }, (element) => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        });

        // Slide animations
        const slideElements = document.querySelectorAll('[data-animation^="slide-"]');
        this.createScrollObserver(slideElements, (element) => {
            const direction = element.dataset.animation.split('-')[1];
            const transform = this.getSlideTransform(direction);
            element.style.opacity = '0';
            element.style.transform = transform;
            element.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        }, (element) => {
            element.style.opacity = '1';
            element.style.transform = 'translate(0, 0)';
        });

        // Scale animations
        const scaleElements = document.querySelectorAll('[data-animation="scale-in"]');
        this.createScrollObserver(scaleElements, (element) => {
            element.style.opacity = '0';
            element.style.transform = 'scale(0.8)';
            element.style.transition = 'all 0.5s ease-out';
        }, (element) => {
            element.style.opacity = '1';
            element.style.transform = 'scale(1)';
        });

        // Stagger animations
        const staggerElements = document.querySelectorAll('[data-animation="stagger"]');
        this.createStaggerAnimation(staggerElements);
    }

    getSlideTransform(direction) {
        const transforms = {
            'left': 'translateX(-50px)',
            'right': 'translateX(50px)',
            'up': 'translateY(-50px)',
            'down': 'translateY(50px)'
        };
        return transforms[direction] || 'translateY(30px)';
    }

    createScrollObserver(elements, setupCallback, animateCallback) {
        if (elements.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const delay = entry.target.dataset.delay || 0;
                    setTimeout(() => {
                        animateCallback(entry.target);
                    }, parseInt(delay));
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        elements.forEach(element => {
            setupCallback(element);
            observer.observe(element);
        });

        this.observers.set('scroll-animations', observer);
    }

    createStaggerAnimation(elements) {
        if (elements.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const children = entry.target.children;
                    Array.from(children).forEach((child, index) => {
                        child.style.opacity = '0';
                        child.style.transform = 'translateY(20px)';
                        child.style.transition = `all 0.6s ease-out ${index * 0.1}s`;
                        
                        setTimeout(() => {
                            child.style.opacity = '1';
                            child.style.transform = 'translateY(0)';
                        }, index * 100);
                    });
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        elements.forEach(element => observer.observe(element));
    }

    initializeHoverEffects() {
        // Card hover effects
        this.addHoverEffect('.glass-card, .team-card, .highlight-card', {
            enter: (element) => {
                element.style.transform = 'translateY(-10px) scale(1.02)';
                element.style.boxShadow = '0 20px 40px rgba(123, 97, 255, 0.3)';
                element.style.transition = 'all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
            },
            leave: (element) => {
                element.style.transform = 'translateY(0) scale(1)';
                element.style.boxShadow = '0 10px 30px rgba(0,0,0,0.2)';
            }
        });

        // Button hover effects
        this.addHoverEffect('.btn-gradient', {
            enter: (element) => {
                element.style.transform = 'translateY(-2px)';
                element.style.boxShadow = '0 10px 25px rgba(123, 97, 255, 0.4)';
                element.style.transition = 'all 0.2s ease';
            },
            leave: (element) => {
                element.style.transform = 'translateY(0)';
                element.style.boxShadow = '0 5px 15px rgba(123, 97, 255, 0.2)';
            }
        });

        // Navigation link effects
        this.addHoverEffect('.nav-link', {
            enter: (element) => {
                element.style.transform = 'scale(1.05)';
                element.style.transition = 'all 0.2s ease';
                element.style.color = '#7b61ff';
            },
            leave: (element) => {
                element.style.transform = 'scale(1)';
                element.style.color = '';
            }
        });

        // Image hover effects
        this.addHoverEffect('.hover-zoom img', {
            enter: (element) => {
                element.style.transform = 'scale(1.1)';
                element.style.transition = 'transform 0.3s ease';
            },
            leave: (element) => {
                element.style.transform = 'scale(1)';
            }
        });
    }

    addHoverEffect(selector, effects) {
        document.addEventListener('mouseover', (e) => {
            const element = e.target.closest(selector);
            if (element && effects.enter) {
                effects.enter(element);
            }
        });

        document.addEventListener('mouseout', (e) => {
            const element = e.target.closest(selector);
            if (element && effects.leave) {
                effects.leave(element);
            }
        });
    }

    initializeParallaxEffects() {
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        
        if (parallaxElements.length > 0) {
            let ticking = false;
            
            const updateParallax = () => {
                const scrollTop = window.pageYOffset;
                
                parallaxElements.forEach(element => {
                    const speed = parseFloat(element.dataset.parallax) || 0.5;
                    const yPos = -(scrollTop * speed);
                    element.style.transform = `translateY(${yPos}px)`;
                });
                
                ticking = false;
            };

            window.addEventListener('scroll', () => {
                if (!ticking) {
                    requestAnimationFrame(updateParallax);
                    ticking = true;
                }
            });
        }
    }

    initializeLoadingAnimations() {
        // Shimmer effect for loading states
        this.createShimmerEffect();
        
        // Pulse animation for loading buttons
        this.createPulseAnimation();
        
        // Spinner animations
        this.createSpinnerAnimations();
    }

    createShimmerEffect() {
        const style = document.createElement('style');
        style.textContent = `
            .shimmer {
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: shimmer 2s infinite;
            }
            
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
        `;
        document.head.appendChild(style);
    }

    createPulseAnimation() {
        const style = document.createElement('style');
        style.textContent = `
            .pulse {
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        `;
        document.head.appendChild(style);
    }

    createSpinnerAnimations() {
        const style = document.createElement('style');
        style.textContent = `
            .spinner {
                animation: spin 1s linear infinite;
            }
            
            .spinner-slow {
                animation: spin 2s linear infinite;
            }
            
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            .bounce {
                animation: bounce 1s infinite;
            }
            
            @keyframes bounce {
                0%, 20%, 53%, 80%, 100% {
                    animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
                    transform: translate3d(0,0,0);
                }
                40%, 43% {
                    animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
                    transform: translate3d(0, -30px, 0);
                }
                70% {
                    animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
                    transform: translate3d(0, -15px, 0);
                }
                90% { transform: translate3d(0,-4px,0); }
            }
        `;
        document.head.appendChild(style);
    }

    // Floating elements animation
    initializeFloatingElements() {
        const floatingElements = document.querySelectorAll('[data-float]');
        
        floatingElements.forEach((element, index) => {
            const speed = element.dataset.float || 3;
            const delay = index * 0.5;
            
            element.style.animation = `float ${speed}s ease-in-out ${delay}s infinite`;
        });

        // Create floating animation keyframes
        this.createFloatingAnimation();
    }

    createFloatingAnimation() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0%, 100% {
                    transform: translateY(0px);
                }
                50% {
                    transform: translateY(-20px);
                }
            }
            
            .float-gentle {
                animation: float-gentle 4s ease-in-out infinite;
            }
            
            @keyframes float-gentle {
                0%, 100% {
                    transform: translateY(0px) rotate(0deg);
                }
                33% {
                    transform: translateY(-10px) rotate(1deg);
                }
                66% {
                    transform: translateY(5px) rotate(-1deg);
                }
            }
        `;
        document.head.appendChild(style);
    }

    initializeTypewriterEffect() {
        const typewriterElements = document.querySelectorAll('[data-typewriter]');
        
        typewriterElements.forEach(element => {
            const text = element.textContent;
            const speed = parseInt(element.dataset.speed) || 100;
            
            element.textContent = '';
            element.style.borderRight = '2px solid #7b61ff';
            
            this.typeWriter(element, text, speed);
        });
    }

    typeWriter(element, text, speed, index = 0) {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            setTimeout(() => this.typeWriter(element, text, speed, index + 1), speed);
        } else {
            // Blinking cursor effect
            setTimeout(() => {
                element.style.borderRight = element.style.borderRight === 'none' ? '2px solid #7b61ff' : 'none';
                setTimeout(() => this.typeWriter(element, text, speed, index), 500);
            }, 1000);
        }
    }

    initializeProgressBars() {
        const progressBars = document.querySelectorAll('[data-progress]');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const progressBar = entry.target;
                    const targetWidth = progressBar.dataset.progress;
                    
                    progressBar.style.width = '0%';
                    progressBar.style.transition = 'width 2s ease-out';
                    
                    setTimeout(() => {
                        progressBar.style.width = targetWidth + '%';
                    }, 100);
                    
                    observer.unobserve(progressBar);
                }
            });
        }, { threshold: 0.5 });

        progressBars.forEach(bar => observer.observe(bar));
    }

    initializeCounterAnimations() {
        const counters = document.querySelectorAll('[data-counter]');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const counter = entry.target;
                    const target = parseInt(counter.dataset.counter);
                    const duration = parseInt(counter.dataset.duration) || 2000;
                    
                    this.animateCounter(counter, 0, target, duration);
                    observer.unobserve(counter);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(counter => observer.observe(counter));
    }

    animateCounter(element, start, end, duration) {
        const startTime = performance.now();
        
        const updateCounter = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentValue = Math.floor(start + (end - start) * this.easeOutCubic(progress));
            element.textContent = currentValue.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        };
        
        requestAnimationFrame(updateCounter);
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    initializeParticleEffects() {
        const particleContainers = document.querySelectorAll('[data-particles]');
        
        particleContainers.forEach(container => {
            this.createParticleSystem(container);
        });
    }

    createParticleSystem(container) {
        const particleCount = parseInt(container.dataset.particles) || 50;
        const particles = [];
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: 2px;
                height: 2px;
                background: rgba(123, 97, 255, 0.6);
                border-radius: 50%;
                pointer-events: none;
            `;
            
            container.appendChild(particle);
            particles.push({
                element: particle,
                x: Math.random() * container.offsetWidth,
                y: Math.random() * container.offsetHeight,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2
            });
        }
        
        this.animateParticles(particles, container);
    }

    animateParticles(particles, container) {
        const animate = () => {
            particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                
                if (particle.x < 0 || particle.x > container.offsetWidth) particle.vx *= -1;
                if (particle.y < 0 || particle.y > container.offsetHeight) particle.vy *= -1;
                
                particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
            });
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }

    // Utility methods
    addToQueue(animation) {
        this.animationQueue.push(animation);
        if (!this.isAnimating) {
            this.processQueue();
        }
    }

    processQueue() {
        if (this.animationQueue.length === 0) {
            this.isAnimating = false;
            return;
        }
        
        this.isAnimating = true;
        const animation = this.animationQueue.shift();
        animation.call(this);
    }

    // Public methods for external use
    fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.transition = `opacity ${duration}ms ease`;
        
        requestAnimationFrame(() => {
            element.style.opacity = '1';
        });
    }

    fadeOut(element, duration = 300) {
        element.style.transition = `opacity ${duration}ms ease`;
        element.style.opacity = '0';
        
        setTimeout(() => {
            element.style.display = 'none';
        }, duration);
    }

    slideDown(element, duration = 300) {
        element.style.overflow = 'hidden';
        element.style.height = '0';
        element.style.transition = `height ${duration}ms ease`;
        
        const height = element.scrollHeight;
        requestAnimationFrame(() => {
            element.style.height = height + 'px';
        });
        
        setTimeout(() => {
            element.style.height = '';
            element.style.overflow = '';
        }, duration);
    }

    slideUp(element, duration = 300) {
        element.style.overflow = 'hidden';
        element.style.height = element.scrollHeight + 'px';
        element.style.transition = `height ${duration}ms ease`;
        
        requestAnimationFrame(() => {
            element.style.height = '0';
        });
        
        setTimeout(() => {
            element.style.display = 'none';
            element.style.height = '';
            element.style.overflow = '';
        }, duration);
    }

    cleanup() {
        // Clean up observers
        this.observers.forEach(observer => observer.disconnect());
        this.observers.clear();
        
        // Clear animation queue
        this.animationQueue = [];
        this.isAnimating = false;
        
        console.log('Animation Controller cleaned up');
    }
}

// Initialize animation controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.animationController = new AnimationController();
    window.animationController.init();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationController;
}
