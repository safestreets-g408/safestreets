import React from 'react';
import { View, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { Card, Title, Paragraph, Badge, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const ReportCard = ({ report, onPress }) => {
  const getStatusColor = (status) => {
    switch(status) {
      case 'pending': return '#FFC107';  // Amber
      case 'assigned': return '#2196F3'; // Blue
      case 'in_progress': return '#9C27B0'; // Purple
      case 'completed': return '#4CAF50'; // Green
      default: return '#9E9E9E'; // Grey
    }
  };

  const getFormattedDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  return (
    <TouchableOpacity onPress={onPress} style={styles.cardContainer}>
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.headerRow}>
            <Title style={styles.cardTitle}>{report.damageType}</Title>
            <Badge
              style={[
                styles.statusBadge,
                { backgroundColor: getStatusColor(report.status) }
              ]}
            >
              {report.status.replace('_', ' ')}
            </Badge>
          </View>
          
          <View style={styles.locationRow}>
            <MaterialCommunityIcons name="map-marker" size={16} color="#666" />
            <Paragraph style={styles.locationText}>{report.location}</Paragraph>
          </View>
          
          <Paragraph style={styles.dateText}>
            Reported: {getFormattedDate(report.createdAt)}
          </Paragraph>
          
          {report.imageUrl && (
            <Image
              source={{ uri: report.imageUrl }}
              style={styles.image}
              resizeMode="cover"
            />
          )}
          
          <View style={styles.tagsContainer}>
            <Chip style={styles.chipTag} textStyle={styles.chipText}>
              {report.severity}
            </Chip>
            <Chip style={styles.chipTag} textStyle={styles.chipText}>
              {report.category}
            </Chip>
          </View>
        </Card.Content>
      </Card>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  cardContainer: {
    marginVertical: 8,
    marginHorizontal: 16,
  },
  card: {
    borderRadius: 10,
    elevation: 2,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    flex: 1,
  },
  statusBadge: {
    marginLeft: 8,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  locationText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 4,
    flex: 1,
  },
  dateText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  image: {
    height: 120,
    borderRadius: 8,
    marginVertical: 8,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  chipTag: {
    marginRight: 8,
    marginTop: 4,
    backgroundColor: '#E0E0E0',
  },
  chipText: {
    fontSize: 12,
  },
});

export default ReportCard;
