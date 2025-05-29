/**
 * Data Loader Utility for CSM-SR Project
 * Handles loading and processing of JSON data files
 */

class DataLoader {
    static async loadJSON(filePath) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error loading JSON from ${filePath}:`, error);
            return null;
        }
    }

    static async loadTeamData() {
        const data = await this.loadJSON('data/team.json');
        if (data) {
            console.log('Team data loaded successfully:', data);
        }
        return data;
    }

    static async loadPublicationsData() {
        const data = await this.loadJSON('data/publications.json');
        if (data) {
            console.log('Publications data loaded successfully:', data);
        }
        return data;
    }

    static async loadResultsData() {
        const data = await this.loadJSON('data/results.json');
        if (data) {
            console.log('Results data loaded successfully:', data);
        }
        return data;
    }

    static async loadConfigData() {
        const data = await this.loadJSON('data/config.json');
        if (data) {
            console.log('Config data loaded successfully:', data);
        }
        return data;
    }

    // Generic method to load any data file
    static async loadData(fileName) {
        return await this.loadJSON(`data/${fileName}`);
    }

    // Method to preload all data files
    static async preloadAllData() {
        const dataFiles = ['team.json', 'publications.json', 'results.json', 'config.json'];
        const loadPromises = dataFiles.map(file => this.loadData(file));
        
        try {
            const results = await Promise.allSettled(loadPromises);
            const loadedData = {};
            
            results.forEach((result, index) => {
                const fileName = dataFiles[index].replace('.json', '');
                if (result.status === 'fulfilled' && result.value) {
                    loadedData[fileName] = result.value;
                } else {
                    console.warn(`Failed to load ${dataFiles[index]}`);
                    loadedData[fileName] = null;
                }
            });
            
            return loadedData;
        } catch (error) {
            console.error('Error preloading data:', error);
            return {};
        }
    }

    // Utility method to handle image loading with fallbacks
    static handleImageLoad(imgElement, fallbackSrc = 'assets/images/placeholder.jpg') {
        return new Promise((resolve) => {
            imgElement.onload = () => resolve(true);
            imgElement.onerror = () => {
                imgElement.src = fallbackSrc;
                imgElement.onload = () => resolve(false);
                imgElement.onerror = () => resolve(false);
            };
        });
    }

    // Method to validate data structure
    static validateTeamData(data) {
        if (!data || !Array.isArray(data.members)) {
            return false;
        }
        
        return data.members.every(member => 
            member.name && 
            member.role && 
            member.bio
        );
    }

    static validatePublicationsData(data) {
        if (!data || !Array.isArray(data.publications)) {
            return false;
        }
        
        return data.publications.every(pub => 
            pub.title && 
            pub.authors && 
            pub.year
        );
    }

    // Cache management
    static cache = new Map();
    
    static async loadWithCache(filePath) {
        if (this.cache.has(filePath)) {
            console.log(`Loading ${filePath} from cache`);
            return this.cache.get(filePath);
        }
        
        const data = await this.loadJSON(filePath);
        if (data) {
            this.cache.set(filePath, data);
        }
        return data;
    }

    static clearCache() {
        this.cache.clear();
        console.log('Data cache cleared');
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.DataLoader = DataLoader;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataLoader;
}
