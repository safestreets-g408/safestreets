import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Avatar } from 'react-native-paper';

const HeaderComponent = ({ fieldWorker, cityStats }) => {
  return (
    <LinearGradient
      colors={['#003366', '#004080']}
      style={styles.header}
      start={{x: 0, y: 0}}
      end={{x: 1, y: 1}}
    >
      <Animatable.View 
        animation="fadeIn" 
        duration={800}
        style={styles.headerContent}
      >
        <View>
          <Animatable.Text 
            animation="fadeInDown" 
            duration={800} 
            style={styles.greeting}
          >
            Hello, {fieldWorker?.name?.split(' ')[0] || 'Field Worker'}
          </Animatable.Text>
          <Animatable.Text 
            animation="fadeInDown" 
            duration={800} 
            delay={200}
            style={styles.subGreeting}
          >
            {fieldWorker?.specialization || 'Road Maintenance'} • {fieldWorker?.region || 'Your Region'}
          </Animatable.Text>
        </View>
        
        <Animatable.View 
          animation="fadeIn"
          duration={1000}
          style={styles.headerActions}
        >
          <View style={styles.avatarContainer}>
            {fieldWorker?.profileImage ? (
              <Avatar.Image 
                size={44} 
                source={{ uri: fieldWorker.profileImage }} 
                style={styles.avatar}
              />
            ) : (
              <Avatar.Text 
                size={44} 
                label={fieldWorker?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'} 
                style={styles.avatar}
                color="#fff"
                backgroundColor="rgba(255,255,255,0.2)" 
              />
            )}
            <View style={styles.onlineIndicator} />
          </View>
        </Animatable.View>
      </Animatable.View>
      
      <View style={styles.headerStatsContainer}>
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.reportsThisWeek || 0}</Text>
          <Text style={styles.headerStatLabel}>New Reports</Text>
        </View>
        <View style={styles.headerStatDivider} />
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.pendingIssues || 0}</Text>
          <Text style={styles.headerStatLabel}>Pending</Text>
        </View>
        <View style={styles.headerStatDivider} />
        <View style={styles.headerStat}>
          <Text style={styles.headerStatNumber}>{cityStats.repairsCompleted || 0}</Text>
          <Text style={styles.headerStatLabel}>Completed</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  header: {
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 24,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    elevation: 8,
    shadowColor: 'rgba(0,0,0,0.4)',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    marginTop: Platform.OS === 'ios' ? -48 : 0,
    zIndex: 10,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    elevation: 4,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  onlineIndicator: {
    width: 12,
    height: 12,
    backgroundColor: '#4CAF50',
    borderRadius: 6,
    borderWidth: 2,
    borderColor: 'white',
    position: 'absolute',
    bottom: 0,
    right: 0,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    letterSpacing: 0.5,
  },
  subGreeting: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.85)',
    marginTop: 4,
    letterSpacing: 0.3,
  },
  headerStatsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 16,
    paddingVertical: 12,
    marginHorizontal: 4,
    marginTop: 8,
  },
  headerStat: {
    alignItems: 'center',
  },
  headerStatNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  headerStatLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  headerStatDivider: {
    width: 1,
    height: '60%',
    backgroundColor: 'rgba(255,255,255,0.3)',
    alignSelf: 'center',
  }
});

export default HeaderComponent;
