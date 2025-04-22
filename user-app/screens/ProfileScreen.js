import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Image, 
  ScrollView, 
  TouchableOpacity,
  Alert,
  ImageBackground
} from 'react-native';
import { Card, Title, Button, TextInput, Avatar, Divider, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';

const ProfileScreen = ({ navigation }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    address: ''
  });

  // Mock user data
  const mockUser = {
    id: 1,
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '(555) 123-4567',
    address: '123 Main St, San Francisco, CA',
    avatarUrl: 'https://via.placeholder.com/150',
    reportCount: 12,
    joinDate: '2023-01-15'
  };

  useEffect(() => {
    // Simulate API call to fetch user profile
    const fetchUserProfile = async () => {
      try {
        // Simulate network delay
        setTimeout(() => {
          setUser(mockUser);
          setFormData({
            name: mockUser.name,
            email: mockUser.email,
            phone: mockUser.phone,
            address: mockUser.address
          });
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Error fetching user profile:', error);
        Alert.alert('Error', 'Failed to load profile information');
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, []);

  const handleSaveProfile = () => {
    setLoading(true);
    
    // Simulate API call to update profile
    setTimeout(() => {
      setUser({
        ...user,
        ...formData
      });
      setLoading(false);
      setEditing(false);
      Alert.alert('Success', 'Profile updated successfully');
    }, 1000);
  };

  const handleInputChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value
    });
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.loadingText}>Loading your profile...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#4c669f', '#3b5998', '#192f6a']}
        style={styles.headerGradient}
      >
        <View style={styles.profileHeader}>
          <Avatar.Image 
            source={{ uri: user.avatarUrl }} 
            size={120} 
            style={styles.avatar}
          />
          <View style={styles.profileInfo}>
            <Text style={styles.profileName}>{user.name}</Text>
            <View style={styles.badgeContainer}>
              <View style={styles.badge}>
                <Text style={styles.badgeText}>Premium</Text>
              </View>
            </View>
            <Text style={styles.joinDate}>Member since {new Date(user.joinDate).toLocaleDateString()}</Text>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{user.reportCount}</Text>
                <Text style={styles.statLabel}>Reports</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>4.8</Text>
                <Text style={styles.statLabel}>Rating</Text>
              </View>
            </View>
          </View>
        </View>
      </LinearGradient>

      <Card style={styles.detailsCard}>
        <Card.Content>
          <View style={styles.sectionTitleContainer}>
            <Title style={styles.sectionTitle}>Personal Information</Title>
            {!editing && (
              <IconButton 
                icon="pencil" 
                size={20} 
                color="#4c669f" 
                onPress={() => setEditing(true)}
                style={styles.editIcon}
              />
            )}
          </View>
          
          {editing ? (
            <View style={styles.formContainer}>
              <TextInput
                label="Full Name"
                value={formData.name}
                onChangeText={(text) => handleInputChange('name', text)}
                style={styles.input}
                mode="outlined"
                theme={{ colors: { primary: '#4c669f' } }}
                left={<TextInput.Icon name="account" color="#4c669f" />}
              />
              <TextInput
                label="Email"
                value={formData.email}
                onChangeText={(text) => handleInputChange('email', text)}
                style={styles.input}
                mode="outlined"
                keyboardType="email-address"
                theme={{ colors: { primary: '#4c669f' } }}
                left={<TextInput.Icon name="email" color="#4c669f" />}
              />
              <TextInput
                label="Phone"
                value={formData.phone}
                onChangeText={(text) => handleInputChange('phone', text)}
                style={styles.input}
                mode="outlined"
                keyboardType="phone-pad"
                theme={{ colors: { primary: '#4c669f' } }}
                left={<TextInput.Icon name="phone" color="#4c669f" />}
              />
              <TextInput
                label="Address"
                value={formData.address}
                onChangeText={(text) => handleInputChange('address', text)}
                style={styles.input}
                mode="outlined"
                multiline
                theme={{ colors: { primary: '#4c669f' } }}
                left={<TextInput.Icon name="map-marker" color="#4c669f" />}
              />
              
              <View style={styles.buttonContainer}>
                <Button 
                  mode="contained" 
                  onPress={handleSaveProfile} 
                  style={styles.saveButton}
                  icon="content-save"
                >
                  Save Changes
                </Button>
                <Button 
                  mode="outlined" 
                  onPress={() => setEditing(false)} 
                  style={styles.cancelButton}
                  icon="close"
                >
                  Cancel
                </Button>
              </View>
            </View>
          ) : (
            <View style={styles.infoContainer}>
              <View style={styles.infoRow}>
                <View style={styles.infoIconContainer}>
                  <IconButton icon="account" size={24} color="#4c669f" />
                </View>
                <View style={styles.infoTextContainer}>
                  <Text style={styles.infoLabel}>Name</Text>
                  <Text style={styles.infoValue}>{user.name}</Text>
                </View>
              </View>
              <Divider style={styles.divider} />
              
              <View style={styles.infoRow}>
                <View style={styles.infoIconContainer}>
                  <IconButton icon="email" size={24} color="#4c669f" />
                </View>
                <View style={styles.infoTextContainer}>
                  <Text style={styles.infoLabel}>Email</Text>
                  <Text style={styles.infoValue}>{user.email}</Text>
                </View>
              </View>
              <Divider style={styles.divider} />
              
              <View style={styles.infoRow}>
                <View style={styles.infoIconContainer}>
                  <IconButton icon="phone" size={24} color="#4c669f" />
                </View>
                <View style={styles.infoTextContainer}>
                  <Text style={styles.infoLabel}>Phone</Text>
                  <Text style={styles.infoValue}>{user.phone}</Text>
                </View>
              </View>
              <Divider style={styles.divider} />
              
              <View style={styles.infoRow}>
                <View style={styles.infoIconContainer}>
                  <IconButton icon="map-marker" size={24} color="#4c669f" />
                </View>
                <View style={styles.infoTextContainer}>
                  <Text style={styles.infoLabel}>Address</Text>
                  <Text style={styles.infoValue}>{user.address}</Text>
                </View>
              </View>
            </View>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.actionsCard}>
        <LinearGradient
          colors={['#f7f7f7', '#eaeaea']}
          style={styles.actionsGradient}
        >
          <Card.Content>
            <Title style={styles.actionTitle}>Quick Actions</Title>
            <TouchableOpacity 
              style={styles.actionButton}
              onPress={() => navigation.navigate('Reports')}
            >
              <LinearGradient
                colors={['#4c669f', '#3b5998']}
                style={styles.actionGradient}
              >
                <IconButton icon="file-document" color="#fff" size={24} />
                <Text style={styles.actionText}>View My Reports</Text>
              </LinearGradient>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.actionButton}
              onPress={() => Alert.alert('Coming Soon', 'This feature is not yet available.')}
            >
              <LinearGradient
                colors={['#4c669f', '#3b5998']}
                style={styles.actionGradient}
              >
                <IconButton icon="cog" color="#fff" size={24} />
                <Text style={styles.actionText}>Account Settings</Text>
              </LinearGradient>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.actionButton, styles.logoutButton]}
              onPress={() => Alert.alert('Logout', 'Are you sure you want to logout?')}
            >
              <LinearGradient
                colors={['#e74c3c', '#c0392b']}
                style={styles.actionGradient}
              >
                <IconButton icon="logout" color="#fff" size={24} />
                <Text style={styles.actionText}>Logout</Text>
              </LinearGradient>
            </TouchableOpacity>
          </Card.Content>
        </LinearGradient>
      </Card>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f2f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f2f5',
  },
  loadingAnimation: {
    width: 150,
    height: 150,
    marginBottom: 20,
  },
  loadingText: {
    fontSize: 18,
    color: '#4c669f',
    fontWeight: '500',
  },
  headerGradient: {
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  profileHeader: {
    flexDirection: 'row',
    padding: 24,
    alignItems: 'center',
  },
  avatar: {
    marginRight: 20,
    backgroundColor: '#e0e0e0',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.6)',
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 4,
  },
  badgeContainer: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  badge: {
    backgroundColor: 'rgba(255,255,255,0.3)',
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 12,
    marginRight: 8,
  },
  badgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  joinDate: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 12,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 12,
    padding: 10,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.3)',
    marginHorizontal: 10,
  },
  detailsCard: {
    margin: 16,
    marginTop: -20,
    elevation: 4,
    borderRadius: 12,
    backgroundColor: 'white',
  },
  sectionTitleContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4c669f',
  },
  editIcon: {
    backgroundColor: 'rgba(76, 102, 159, 0.1)',
    margin: 0,
  },
  infoContainer: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    padding: 8,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  infoIconContainer: {
    marginRight: 8,
  },
  infoTextContainer: {
    flex: 1,
  },
  infoLabel: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 2,
  },
  infoValue: {
    fontSize: 16,
    color: '#2c3e50',
    fontWeight: '500',
  },
  divider: {
    marginVertical: 4,
    height: 1,
    backgroundColor: '#e0e0e0',
  },
  formContainer: {
    marginTop: 8,
  },
  input: {
    marginBottom: 16,
    backgroundColor: 'white',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  saveButton: {
    flex: 1,
    marginRight: 8,
    backgroundColor: '#4c669f',
    borderRadius: 8,
    paddingVertical: 6,
  },
  cancelButton: {
    flex: 1,
    marginLeft: 8,
    borderColor: '#4c669f',
    borderRadius: 8,
  },
  actionsCard: {
    margin: 16,
    marginTop: 8,
    marginBottom: 24,
    elevation: 4,
    borderRadius: 12,
    overflow: 'hidden',
  },
  actionsGradient: {
    padding: 8,
  },
  actionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4c669f',
    marginBottom: 16,
  },
  actionButton: {
    marginVertical: 8,
    borderRadius: 12,
    overflow: 'hidden',
    elevation: 2,
  },
  actionGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  actionText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    flex: 1,
  },
  logoutButton: {
    marginTop: 16,
  }
});

export default ProfileScreen;
