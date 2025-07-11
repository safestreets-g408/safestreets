// RootNavigation.js
import { createRef } from 'react';

/**
 * Navigation reference to allow navigation from outside of components
 * This is particularly useful for contexts, services, and utils
 */
export const navigationRef = createRef();

/**
 * Navigate to a route
 * @param {string} name - The route name
 * @param {Object} params - The parameters to pass to the route
 */
export const navigate = (name, params) => {
  if (navigationRef.current) {
    navigationRef.current.navigate(name, params);
  } else {
    // Queue navigation for when navigation is ready
    setTimeout(() => {
      if (navigationRef.current) {
        navigationRef.current.navigate(name, params);
      }
    }, 500);
  }
};

/**
 * Go back in navigation stack
 */
export const goBack = () => {
  if (navigationRef.current) {
    navigationRef.current.goBack();
  }
};

/**
 * Reset the navigation state
 * @param {Object} state - The navigation state to reset to
 */
export const reset = (state) => {
  if (navigationRef.current) {
    navigationRef.current.reset(state);
  }
};

/**
 * Check if the navigation is ready
 * @returns {boolean} - True if navigation is ready
 */
export const isReady = () => {
  return navigationRef.current?.isReady() || false;
};

export default {
  navigationRef,
  navigate,
  goBack,
  reset,
  isReady,
};
