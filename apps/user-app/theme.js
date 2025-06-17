import { DefaultTheme } from 'react-native-paper';

// Define application theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#003366',
    secondary: '#2196f3',
    accent: '#03DAC6',
    background: '#F8F9FA',
    surface: '#FFFFFF',
    error: '#B00020',
    text: '#212121',
    placeholder: '#9E9E9E',
    disabled: '#BDBDBD',
  },
  roundness: 8,
};

export default theme;
