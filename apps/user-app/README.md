# SafeStreets Field Worker Mobile App

## Project Structure

The SafeStreets Field Worker Mobile App has been restructured to follow a clean, modular architecture that makes the codebase more maintainable and easier to extend.

### Directory Structure

```
user-app/
├── assets/              # Static assets (images, fonts, etc.)
├── components/          # Reusable UI components
│   ├── home/            # Home screen specific components
│   ├── layout/          # Layout components (AppProvider, Screen)
│   ├── navigation/      # Navigation components (AppNavigator, MainTabs)
│   └── ui/              # Shared UI components (LoadingSpinner, ErrorFallback)
├── context/             # React Context providers (AuthContext)
├── hooks/               # Custom React hooks
├── screens/             # App screens
├── utils/               # Utility functions and API services
├── App.js               # Main app component
├── app.json             # Expo configuration
├── config.js            # Global app configuration
├── index.js             # Entry point
└── theme.js             # App theme configuration
```

### Key Components

- **components/layout/AppProvider.js**: Central provider that wraps the app with all required context providers
- **components/navigation/AppNavigator.js**: Main navigation container with authentication flow
- **components/navigation/MainTabs.js**: Bottom tab navigation for authenticated users
- **context/AuthContext.js**: Authentication state management
- **utils/**: Utility modules and API services
  - **auth.js**: Authentication utilities and API calls
  - **reports.js**: Report management API calls
  - **tasks.js**: Task management API calls
  - **formatters.js**: Data formatting helpers

### Code Organization Principles

1. **Component-Based Architecture**: Each UI element is a reusable component
2. **Separation of Concerns**: Each module has a single responsibility
3. **Clean API Boundaries**: Utility functions handle API calls and data transformation
4. **DRY Code**: Common functionality is extracted into shared components and hooks
5. **Consistent Styling**: Theme is centrally defined and reused throughout the app

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Update API configuration in `config.js`:
   ```javascript
   export const API_BASE_URL = 'http://your-api-url.com/api';
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Development Guidelines

1. Follow the established directory structure
2. Create reusable components in the appropriate subdirectories
3. Use the provided context and utility functions
4. Implement proper error handling
5. Document complex functions and components

## Testing

Run tests with:
```bash
npm test
```

## Building for Production

Build for production with:
```bash
npm run build
```
