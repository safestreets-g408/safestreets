import React from 'react';
import { StyleSheet, TouchableOpacity, Text, ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from 'react-native-paper';


const GradientButton = ({
  title,
  onPress,
  style,
  textStyle,
  loading = false,
  disabled = false,
  mode = 'primary',
  size = 'medium',
  outlined = false,
  icon = null,
}) => {
  const theme = useTheme();
  
  // Define color pairs for different button modes - matching admin portal gradients
  const gradientColors = theme.colors.gradient || {
    primary: ['#667eea', '#764ba2'], // Beautiful purple-blue gradient from admin theme
    secondary: [theme.colors.secondary, theme.colors.secondaryDark],
    success: [theme.colors.success, theme.colors.successDark],
    error: [theme.colors.error, theme.colors.errorDark],
    warning: [theme.colors.warning, theme.colors.warningDark],
    info: [theme.colors.info, theme.colors.infoDark],
    blue: ['#60a5fa', '#3b82f6'],
    purple: ['#667eea', '#764ba2'],
  };
  
  // Define size styles
  const sizeStyles = {
    small: { paddingVertical: 6, paddingHorizontal: 12 },
    medium: { paddingVertical: 10, paddingHorizontal: 16 },
    large: { paddingVertical: 14, paddingHorizontal: 24 },
  };
  
  const textSizeStyles = {
    small: { fontSize: 12 },
    medium: { fontSize: 14 },
    large: { fontSize: 16 },
  };
  
  // Get colors for current mode
  const colors = gradientColors[mode] || gradientColors.primary;
  const textColor = outlined ? colors[0] : '#FFFFFF';
  
  // Set up container and text styles
  const containerStyle = [
    styles.button,
    sizeStyles[size] || sizeStyles.medium,
    { borderRadius: theme.roundness * 1.5 }, // More rounded corners match admin portal
    disabled && styles.disabled,
    style,
  ];
  
  const buttonTextStyle = [
    styles.text,
    textSizeStyles[size] || textSizeStyles.medium,
    { color: textColor },
    disabled && styles.disabledText,
    textStyle,
  ];
  
  // Handle outlined mode
  if (outlined) {
    return (
      <TouchableOpacity
        style={[containerStyle, { borderColor: colors[0], borderWidth: 1 }]}
        onPress={onPress}
        disabled={disabled || loading}
        activeOpacity={0.7}
      >
        {loading ? (
          <ActivityIndicator color={colors[0]} size="small" />
        ) : (
          <>
            {icon && icon}
            <Text style={buttonTextStyle}>{title}</Text>
          </>
        )}
      </TouchableOpacity>
    );
  }
  
  // Regular gradient button
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || loading}
      style={containerStyle}
      activeOpacity={0.7}
    >
      <LinearGradient
        colors={colors}
        style={styles.gradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
      >
        {loading ? (
          <ActivityIndicator color="#FFFFFF" size="small" />
        ) : (
          <>
            {icon && icon}
            <Text style={buttonTextStyle}>{title}</Text>
          </>
        )}
      </LinearGradient>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: 8,
  },
  gradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  text: {
    fontWeight: '600',
    textAlign: 'center',
    marginLeft: 8,
  },
  disabled: {
    opacity: 0.5,
  },
  disabledText: {
    opacity: 0.8,
  },
});

export default GradientButton;
