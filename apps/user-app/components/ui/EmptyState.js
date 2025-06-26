import React from 'react';
import { View, StyleSheet, Dimensions, ActivityIndicator } from 'react-native';
import { useTheme, Text } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';


const EmptyState = ({
  title = 'Nothing to show',
  message = 'There are no items to display right now.',
  icon = 'information-outline',
  action,
  loading = false,
  style,
}) => {
  const theme = useTheme();
  const screenWidth = Dimensions.get('window').width;
  
  // If loading, show spinner
  if (loading) {
    return (
      <View style={[styles.container, style]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.message, { color: theme.colors.textSecondary, marginTop: 16 }]}>
          Loading...
        </Text>
      </View>
    );
  }
  
  return (
    <View style={[styles.container, style]}>
      <View style={styles.iconContainer}>
        <LinearGradient
          colors={[theme.colors.primaryLight + '40', theme.colors.primary + '40']}
          style={styles.iconBackground}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <MaterialCommunityIcons
            name={icon}
            size={screenWidth * 0.15}
            color={theme.colors.primary}
          />
        </LinearGradient>
      </View>
      
      <Text style={[styles.title, { color: theme.colors.text }]}>
        {title}
      </Text>
      
      <Text style={[styles.message, { color: theme.colors.textSecondary }]}>
        {message}
      </Text>
      
      {action && (
        <TouchableOpacity 
          style={[styles.actionButton, { backgroundColor: theme.colors.primary }]} 
          onPress={action.onPress}
        >
          <Text style={styles.actionText}>{action.label}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  iconContainer: {
    marginBottom: 20,
  },
  iconBackground: {
    width: 90,
    height: 90,
    borderRadius: 45,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginHorizontal: 24,
    marginBottom: 24,
  },
  actionButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  actionText: {
    color: '#FFF',
    fontWeight: '600',
    fontSize: 16,
  },
});

export default EmptyState;
