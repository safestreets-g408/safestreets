import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Card, Title, Button, TextInput, Avatar, Divider, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';
import { updateFieldWorkerProfile } from '../utils/auth';

const ProfileScreen = ({ navigation }) => {
  const { fieldWorker, logout, updateFieldWorker } = useAuth();
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState({
    name: fieldWorker?.name || '',
    specialization: fieldWorker?.specialization || '',
    region: fieldWorker?.region || '',
    profile: {
      phone: fieldWorker?.profile?.phone || '',
    }
  });

  useEffect(() => {
    if (fieldWorker) {
      setFormData({
        name: fieldWorker.name,
        specialization: fieldWorker.specialization,
        region: fieldWorker.region,
        profile: {
          phone: fieldWorker.profile?.phone || '',
        }
      });
    }
  }, [fieldWorker]);

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Logout', 
          style: 'destructive',
          onPress: async () => {
            await logout();
            navigation.reset({
              index: 0,
              routes: [{ name: 'Login' }],
            });
          }
        }
      ]
    );
  };

  const handleSaveProfile = async () => {
    setLoading(true);
    
    try {
      const updatedFieldWorker = await updateFieldWorkerProfile(formData);
      updateFieldWorker(updatedFieldWorker);
      setEditing(false);
      Alert.alert('Success', 'Profile updated successfully');
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelEdit = () => {
    if (fieldWorker) {
      setFormData({
        name: fieldWorker.name,
        specialization: fieldWorker.specialization,
        region: fieldWorker.region,
        profile: {
          phone: fieldWorker.profile?.phone || '',
        }
      });
    }
    setEditing(false);
  };

  if (!fieldWorker) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading profile...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#003366', '#0066CC']}
        style={styles.headerGradient}
      >
        <View style={styles.header}>
          <Avatar.Text 
            size={80} 
            label={fieldWorker.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'} 
            style={styles.avatar}
            labelStyle={{ fontSize: 24, fontWeight: 'bold' }}
          />
          <Title style={styles.name}>{fieldWorker.name}</Title>
          <Text style={styles.email}>{fieldWorker.email}</Text>
          <Text style={styles.workerId}>ID: {fieldWorker.workerId}</Text>
        </View>
      </LinearGradient>

      <View style={styles.content}>
        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Title>Personal Information</Title>
              <IconButton
                icon={editing ? "close" : "pencil"}
                mode="contained"
                onPress={editing ? handleCancelEdit : () => setEditing(true)}
                style={styles.editButton}
              />
            </View>
            
            <Divider style={styles.divider} />
            
            <View style={styles.infoContainer}>
              <View style={styles.infoItem}>
                <Text style={styles.label}>Full Name</Text>
                {editing ? (
                  <TextInput
                    value={formData.name}
                    onChangeText={(text) => setFormData({ ...formData, name: text })}
                    style={styles.input}
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.value}>{fieldWorker.name}</Text>
                )}
              </View>

              <View style={styles.infoItem}>
                <Text style={styles.label}>Specialization</Text>
                {editing ? (
                  <TextInput
                    value={formData.specialization}
                    onChangeText={(text) => setFormData({ ...formData, specialization: text })}
                    style={styles.input}
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.value}>{fieldWorker.specialization}</Text>
                )}
              </View>

              <View style={styles.infoItem}>
                <Text style={styles.label}>Region</Text>
                {editing ? (
                  <TextInput
                    value={formData.region}
                    onChangeText={(text) => setFormData({ ...formData, region: text })}
                    style={styles.input}
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.value}>{fieldWorker.region}</Text>
                )}
              </View>

              <View style={styles.infoItem}>
                <Text style={styles.label}>Phone Number</Text>
                {editing ? (
                  <TextInput
                    value={formData.profile.phone}
                    onChangeText={(text) => setFormData({ 
                      ...formData, 
                      profile: { ...formData.profile, phone: text }
                    })}
                    style={styles.input}
                    mode="outlined"
                    keyboardType="phone-pad"
                  />
                ) : (
                  <Text style={styles.value}>{fieldWorker.profile?.phone || 'Not set'}</Text>
                )}
              </View>
            </View>

            {editing && (
              <View style={styles.buttonContainer}>
                <Button
                  mode="contained"
                  onPress={handleSaveProfile}
                  loading={loading}
                  disabled={loading}
                  style={styles.saveButton}
                >
                  Save Changes
                </Button>
              </View>
            )}
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Title>Work Statistics</Title>
            <Divider style={styles.divider} />
            
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{fieldWorker.activeAssignments || 0}</Text>
                <Text style={styles.statLabel}>Active Assignments</Text>
              </View>
              
              <View style={styles.statDivider} />
              
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{fieldWorker.profile?.totalReportsHandled || 0}</Text>
                <Text style={styles.statLabel}>Reports Handled</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Title>Account Settings</Title>
            <Divider style={styles.divider} />
            
            <TouchableOpacity 
              style={styles.settingItem}
              onPress={() => Alert.alert('Change Password', 'Contact administrator to change password')}
            >
              <Text style={styles.settingText}>Change Password</Text>
              <IconButton icon="chevron-right" size={20} />
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.settingItem}
              onPress={() => Alert.alert('Notifications', 'Notification settings coming soon')}
            >
              <Text style={styles.settingText}>Notification Settings</Text>
              <IconButton icon="chevron-right" size={20} />
            </TouchableOpacity>
          </Card.Content>
        </Card>

        <Button
          mode="contained"
          onPress={handleLogout}
          style={styles.logoutButton}
          buttonColor="#D32F2F"
          icon="logout"
        >
          Logout
        </Button>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerGradient: {
    paddingBottom: 30,
  },
  header: {
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 20,
  },
  avatar: {
    backgroundColor: '#ffffff',
    marginBottom: 15,
  },
  name: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  email: {
    color: '#E3F2FD',
    fontSize: 16,
    marginBottom: 5,
  },
  workerId: {
    color: '#B3E5FC',
    fontSize: 14,
    fontWeight: '500',
  },
  content: {
    padding: 20,
    marginTop: -15,
  },
  card: {
    marginBottom: 20,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  editButton: {
    margin: 0,
  },
  divider: {
    marginBottom: 20,
  },
  infoContainer: {
    gap: 15,
  },
  infoItem: {
    marginBottom: 15,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 5,
  },
  value: {
    fontSize: 16,
    color: '#333',
  },
  input: {
    backgroundColor: '#ffffff',
  },
  buttonContainer: {
    marginTop: 20,
  },
  saveButton: {
    backgroundColor: '#003366',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 20,
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#E0E0E0',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#003366',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  settingText: {
    fontSize: 16,
    color: '#333',
  },
  logoutButton: {
    marginTop: 20,
    marginBottom: 40,
  },
});

export default ProfileScreen;
