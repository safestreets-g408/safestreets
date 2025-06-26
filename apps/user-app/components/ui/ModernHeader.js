import React from 'react';
import { View, StyleSheet, Text, TouchableOpacity, StatusBar, Platform } from 'react-native';
import { Avatar, useTheme } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';


const ModernHeader = ({
  title,
  subtitle,
  user,
  actions = [],
  style,
  showAvatar = true,
  useGradient = false,
  gradientColors,
  elevated = true,
  back,
  transparent = false,
}) => {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  
  // Calculate top padding based on safe area
  const topPadding = Platform.OS === 'ios' ? insets.top : StatusBar.currentHeight || 0;
  
  // Default gradient colors
  const defaultGradientColors = transparent 
    ? ['transparent', 'transparent'] 
    : gradientColors || theme.colors.gradient?.primary || ['#003366', '#004080', '#0055a4'];

  const renderContent = () => (
    <View style={[styles.content, useGradient && { paddingTop: topPadding }]}>
      {back && back.visible && (
        <TouchableOpacity
          style={styles.backButton}
          onPress={back.onPress}
          hitSlop={{top: 15, right: 15, bottom: 15, left: 15}}
        >
          <MaterialCommunityIcons
            name="arrow-left"
            size={24}
            color={useGradient ? "#fff" : theme.colors.text}
          />
        </TouchableOpacity>
      )}
    
      <View style={styles.titleContainer}>
        <Text 
          style={[
            styles.title, 
            { color: useGradient ? "#fff" : theme.colors.text },
          ]}
          numberOfLines={1}
        >
          {title}
        </Text>
        {subtitle && (
          <Text 
            style={[
              styles.subtitle, 
              { color: useGradient ? "rgba(255,255,255,0.8)" : theme.colors.textSecondary }
            ]}
          >
            {subtitle}
          </Text>
        )}
      </View>

      <View style={styles.rightContainer}>
        {actions.map((action, index) => (
          <TouchableOpacity
            key={`action-${index}`}
            style={styles.actionButton}
            onPress={action.onPress}
          >
            <MaterialCommunityIcons
              name={action.icon}
              size={24}
              color={useGradient ? "#fff" : theme.colors.primary}
            />
          </TouchableOpacity>
        ))}
        
        {showAvatar && user && (
          <TouchableOpacity style={styles.avatarContainer}>
            {user.avatar ? (
              <Avatar.Image 
                source={{ uri: user.avatar }} 
                size={40}
                style={styles.avatar} 
              />
            ) : (
              <Avatar.Text 
                label={user.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U'} 
                size={40} 
                style={[
                  styles.avatar,
                  { backgroundColor: useGradient ? 'rgba(255,255,255,0.2)' : theme.colors.primary + '30' }
                ]} 
                color={useGradient ? "#fff" : theme.colors.primary}
              />
            )}
            <View style={[
              styles.onlineIndicator, 
              { borderColor: useGradient ? theme.colors.primaryDark : theme.colors.surface }
            ]} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
  
  return useGradient ? (
    <View 
      style={[
        styles.container,
        elevated && styles.elevated,
        style
      ]}
    >
      <StatusBar 
        barStyle="light-content"
        backgroundColor="transparent"
        translucent={true}
      />
      <LinearGradient
        colors={defaultGradientColors}
        start={{x: 0, y: 0}}
        end={{x: 1, y: 1}}
        style={styles.gradient}
      >
        {renderContent()}
      </LinearGradient>
    </View>
  ) : (
    <View 
      style={[
        styles.container,
        { backgroundColor: theme.colors.surface },
        elevated && styles.elevated,
        style
      ]}
    >
      {renderContent()}
    </View>
  );
};

// Helper function to get initials from a name
const getInitials = (name) => {
  if (!name) return '?';
  return name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .substring(0, 2);
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: 16,
    paddingHorizontal: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 14,
    marginTop: 2,
  },
  rightContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  actionButton: {
    marginHorizontal: 8,
  },
  avatar: {
    marginLeft: 12,
  },
});

export default ModernHeader;
