// app.config.js - provides configuration for Expo app
import { getConfig } from '@expo/config';

// Get the base config from app.json
const config = getConfig(__dirname);

// Set the entry point explicitly
if (config?.expo) {
  config.expo.entryPoint = './index.js';
}

export default config;
