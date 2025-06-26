// app.config.js - provides configuration for Expo app
import { ConfigContext } from '@expo/config';
import { readFileSync } from 'fs';
import { join } from 'path';

// Read the app.json file directly to avoid circular dependency
const appJsonPath = join(__dirname, 'app.json');
const rawConfig = JSON.parse(readFileSync(appJsonPath, 'utf8'));

// Add or modify properties as needed
export default ({ config }) => {
  return {
    ...rawConfig,
    expo: {
      ...rawConfig.expo,
      entryPoint: './index.js',
      // Add any additional dynamic configurations here
    }
  };
};
