import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, ScrollView } from 'react-native';
import { useTheme, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ModernCard } from '../ui';
import { getReportImageUrlSync } from '../../utils/imageUtils';

const RecentReportsComponent = ({ reports, navigation, formatTimeAgo, handleQuickStatusUpdate, loading }) => {
  const theme = useTheme();
  
  const renderRecentReportItem = (item) => {
    // Map status to colors based on theme
    const statusColors = {
      completed: theme.colors.success,
      in_progress: theme.colors.warning,
      pending: theme.colors.info
    };
    
    const statusColor = statusColors[item.repairStatus] || theme.colors.info;
    const statusBgColor = statusColor + '20'; // Light background version
    
    // Get the best image URL for this report
    const getImageUrl = (report) => {
      if (!report) return 'https://via.placeholder.com/300';
      
      try {
        // Try different image types in order of preference
        const imageTypes = ['before', 'main', 'thumbnail', 'default'];
        
        for (const type of imageTypes) {
          const url = getReportImageUrlSync(report, type);
          // If we have a valid URL that's not a placeholder, use it
          if (url && !url.includes('placeholder')) {
            return url;
          }
        }
        
        // Fallback to basic function
        return getReportImageUrlSync(report);
      } catch (error) {
        console.error('Error getting image URL in RecentReportsComponent:', error);
        return 'https://via.placeholder.com/300?text=Error';
      }
    };
    
    return (
      <ModernCard
        style={styles.recentReportCard}
        onPress={() => navigation.navigate('ViewReport', { reportId: item._id })}
        elevation="medium"
      >
        <View style={styles.reportImageContainer}>
          <Image 
            source={{ 
              uri: getImageUrl(item),
              headers: {
                'Cache-Control': 'no-store, no-cache',
                'Pragma': 'no-cache'
              }
            }}
            style={styles.recentReportImage}
            defaultSource={require('../../assets/icon.png')}
            onError={(e) => {
              console.log(`Image load error for report ${item._id || item.id}: ${e.nativeEvent?.error || 'Unknown error'}`);
              // Try fallback URL if primary fails
              const fallbackUrl = `https://via.placeholder.com/300x140/4285f4/ffffff?text=${encodeURIComponent(item.damageType || 'Road Damage')}`;
              if (e.target && e.target.source && e.target.source.uri !== fallbackUrl) {
                e.target.source = { uri: fallbackUrl };
              }
            }}
            onLoad={() => console.log(`Image loaded successfully for report ${item._id || item.id}`)}
          />
          <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.7)']}
            style={styles.imageGradient}
          /> 
          <View style={[styles.reportStatusTag, { backgroundColor: statusColor }]}>
            <MaterialCommunityIcons 
              name={
                item.repairStatus === 'completed' ? 'check-circle' : 
                item.repairStatus === 'in_progress' ? 'progress-clock' : 'clock-outline'
              } 
              size={14} 
              color="#fff" 
            />
            <Text style={styles.reportStatusText}>
              {item.repairStatus?.replace('_', ' ') || 'pending'}
            </Text>
          </View>
        </View>
        <View style={styles.recentReportContent}>
          <Text style={[styles.recentReportTitle, { color: theme.colors.text }]} numberOfLines={1}>
            {item.damageType || 'Road Damage'}
          </Text>
          <Text style={[styles.recentReportLocation, { color: theme.colors.textSecondary }]} numberOfLines={1}>
            <MaterialCommunityIcons name="map-marker" size={14} color={theme.colors.textSecondary} />
            {' '}{item.location || 'Unknown Location'}
          </Text>
          <View style={styles.reportFooter}>
            <View style={styles.timeContainer}>
              <MaterialCommunityIcons name="clock-outline" size={14} color={theme.colors.textSecondary} />
              <Text style={[styles.timeText, { color: theme.colors.textSecondary }]}>
                {formatTimeAgo(new Date(item.createdAt))}
              </Text>
            </View>
            
            {item.repairStatus === 'pending' && (
              <TouchableOpacity 
                style={[styles.actionChip, { backgroundColor: theme.colors.primary }]}
                onPress={() => handleQuickStatusUpdate(item._id, 'in_progress')}
              >
                <MaterialCommunityIcons name="play" size={14} color="#fff" />
                <Text style={styles.actionChipText}>Start</Text>
              </TouchableOpacity>
            )}
            
            {item.repairStatus === 'in_progress' && (
              <TouchableOpacity 
                style={[styles.actionChip, { backgroundColor: theme.colors.success }]}
                onPress={() => handleQuickStatusUpdate(item._id, 'completed')}
              >
                <MaterialCommunityIcons name="check" size={14} color="#fff" />
                <Text style={styles.actionChipText}>Complete</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </ModernCard>
    );
  };
  
  return (
    <>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Your Assigned Reports</Text>
        <TouchableOpacity onPress={() => navigation.navigate('Reports')}>
          <Text style={styles.seeAllText}>See All</Text>
        </TouchableOpacity>
      </View>

      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false} 
        style={styles.recentReportsContainer}
        contentContainerStyle={styles.reportsContentContainer}
      >
        {Array.isArray(reports) && reports.slice(0, 3).map(item => (
          <Animatable.View key={item._id} animation="fadeInRight" duration={500} delay={100}>
            {renderRecentReportItem(item)}
          </Animatable.View>
        ))}
        
        {reports.length === 0 && !loading && (
          <View style={styles.noReportsContainer}>
            <MaterialCommunityIcons name="clipboard-list" size={48} color="#ccc" />
            <Text style={styles.noReportsText}>No assignments yet</Text>
          </View>
        )}
        
        <TouchableOpacity 
          style={styles.newReportCard}
          onPress={() => navigation.navigate('Camera')}
        >
          <LinearGradient
            colors={['#1a73e8', '#4285f4']}
            style={styles.newReportGradient}
          >
            <IconButton icon="plus" color="#fff" size={32} />
            <Text style={styles.newReportText}>New Report</Text>
          </LinearGradient>
        </TouchableOpacity>
      </ScrollView>
    </>
  );
};

const styles = StyleSheet.create({
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 12,
    paddingHorizontal: 4
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold'
  },
  seeAllText: {
    color: '#1a73e8',
    fontWeight: '500'
  },
  recentReportsContainer: {
    marginLeft: -16,
    marginRight: -16,
    paddingLeft: 16
  },
  reportsContentContainer: {
    paddingRight: 16
  },
  recentReportCard: {
    width: 280,
    marginRight: 16,
    borderRadius: 16,
    overflow: 'hidden',
    margin: 0,
    padding: 0
  },
  reportImageContainer: {
    position: 'relative',
    height: 140
  },
  recentReportImage: {
    width: '100%',
    height: '100%',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16
  },
  imageGradient: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 40
  },
  reportStatusTag: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    position: 'absolute',
    top: 10,
    right: 10
  },
  reportStatusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
    marginLeft: 4,
    textTransform: 'capitalize'
  },
  recentReportContent: {
    padding: 12
  },
  recentReportTitle: {
    fontSize: 16,
    fontWeight: '600'
  },
  recentReportLocation: {
    fontSize: 13,
    marginTop: 4
  },
  reportFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8
  },
  timeContainer: {
    flexDirection: 'row',
    alignItems: 'center'
  },
  timeText: {
    fontSize: 12,
    marginLeft: 4
  },
  actionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 12
  },
  actionChipText: {
    color: 'white',
    fontSize: 12,
    marginLeft: 4
  },
  newReportCard: {
    width: 140,
    height: 220,
    borderRadius: 16,
    overflow: 'hidden'
  },
  newReportGradient: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center'
  },
  newReportText: {
    color: 'white',
    fontWeight: '600',
    marginTop: 8
  },
  noReportsContainer: {
    width: 280,
    height: 220,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center'
  },
  noReportsText: {
    marginTop: 12,
    color: '#666',
    fontSize: 16
  }
});

export default RecentReportsComponent;
