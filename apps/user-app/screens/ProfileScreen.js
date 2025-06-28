import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity,
  Alert,
  StatusBar,
  Dimensions,
} from 'react-native';
import { 
  Card, 
  Title, 
  Button, 
  TextInput, 
  Avatar, 
  Divider, 
  IconButton,
  useTheme,
  Chip,
  Surface,
  ActivityIndicator
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { updateFieldWorkerProfile } from '../utils/auth';
import { ModernCard, ConsistentHeader } from '../components/ui';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: screenWidth } = Dimensions.get('window');

const ProfileScreen = ({ navigation }) => {
  const { fieldWorker, logout, updateFieldWorker } = useAuth();
  const theme = useTheme();
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
      <View style={[styles.container, styles.centered, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>
          Loading profile...
        </Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primary} />
      
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, theme.colors.primaryDark]}
        style={styles.headerGradient}
      >
        <View style={styles.header}>
          <Animatable.View animation="fadeInUp" delay={200}>
            <Avatar.Text 
              size={100} 
              label={fieldWorker.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'FW'} 
              style={[styles.avatar, { backgroundColor: theme.colors.surface }]}
              labelStyle={{ fontSize: 28, fontWeight: 'bold', color: theme.colors.primary }}
            />
          </Animatable.View>
          
          <Animatable.View animation="fadeInUp" delay={300}>
            <Title style={[styles.name, { color: theme.colors.onPrimary }]}>
              {fieldWorker.name}
            </Title>
            <Text style={[styles.email, { color: theme.colors.onPrimary + 'DD' }]}>
              {fieldWorker.email}
            </Text>
            <Chip 
              mode="flat"
              style={[styles.workerIdChip, { backgroundColor: theme.colors.surface + 'E6' }]}
              textStyle={{ color: theme.colors.primary, fontWeight: '600' }}
            >
              ID: {fieldWorker.workerId}
            </Chip>
          </Animatable.View>
        </View>
      </LinearGradient>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Personal Information Card */}
        <Animatable.View animation="fadeInUp" delay={400}>
          <ModernCard style={styles.card}>
            <View style={styles.cardHeader}>
              <View style={styles.cardTitleSection}>
                <MaterialCommunityIcons 
                  name="account-circle" 
                  size={24} 
                  color={theme.colors.primary} 
                />
                <Title style={[styles.cardTitle, { color: theme.colors.text }]}>
                  Personal Information
                </Title>
              </View>
              <IconButton
                icon={editing ? "close" : "pencil"}
                iconColor={editing ? theme.colors.error : theme.colors.primary}
                size={24}
                onPress={editing ? handleCancelEdit : () => setEditing(true)}
                style={[styles.editButton, { backgroundColor: theme.colors.surface }]}
              />
            </View>
            
            <Divider style={styles.divider} />

            <View style={styles.infoSection}>
              {editing ? (
                <>
                  <TextInput
                    label="Full Name"
                    value={formData.name}
                    onChangeText={(text) => setFormData({ ...formData, name: text })}
                    style={styles.input}
                    theme={{ colors: { primary: theme.colors.primary } }}
                    left={<TextInput.Icon icon="account" />}
                  />
                  
                  <TextInput
                    label="Specialization"
                    value={formData.specialization}
                    onChangeText={(text) => setFormData({ ...formData, specialization: text })}
                    style={styles.input}
                    theme={{ colors: { primary: theme.colors.primary } }}
                    left={<TextInput.Icon icon="tools" />}
                  />
                  
                  <TextInput
                    label="Region"
                    value={formData.region}
                    onChangeText={(text) => setFormData({ ...formData, region: text })}
                    style={styles.input}
                    theme={{ colors: { primary: theme.colors.primary } }}
                    left={<TextInput.Icon icon="map-marker" />}
                  />
                  
                  <TextInput
                    label="Phone Number"
                    value={formData.profile.phone}
                    onChangeText={(text) => setFormData({ 
                      ...formData, 
                      profile: { ...formData.profile, phone: text }
                    })}
                    style={styles.input}
                    theme={{ colors: { primary: theme.colors.primary } }}
                    left={<TextInput.Icon icon="phone" />}
                  />
                  
                  <View style={styles.buttonContainer}>
                    <Button
                      mode="contained"
                      onPress={handleSaveProfile}
                      loading={loading}
                      disabled={loading}
                      style={[styles.saveButton, { backgroundColor: theme.colors.primary }]}
                      contentStyle={styles.buttonContent}
                    >
                      Save Changes
                    </Button>
                  </View>
                </>
              ) : (
                <>
                  <View style={styles.infoRow}>
                    <MaterialCommunityIcons name="account" size={20} color={theme.colors.primary} />
                    <View style={styles.infoContent}>
                      <Text style={[styles.infoLabel, { color: theme.colors.textSecondary }]}>Full Name</Text>
                      <Text style={[styles.infoValue, { color: theme.colors.text }]}>{fieldWorker.name}</Text>
                    </View>
                  </View>

                  <View style={styles.infoRow}>
                    <MaterialCommunityIcons name="tools" size={20} color={theme.colors.primary} />
                    <View style={styles.infoContent}>
                      <Text style={[styles.infoLabel, { color: theme.colors.textSecondary }]}>Specialization</Text>
                      <Text style={[styles.infoValue, { color: theme.colors.text }]}>{fieldWorker.specialization}</Text>
                    </View>
                  </View>

                  <View style={styles.infoRow}>
                    <MaterialCommunityIcons name="map-marker" size={20} color={theme.colors.primary} />
                    <View style={styles.infoContent}>
                      <Text style={[styles.infoLabel, { color: theme.colors.textSecondary }]}>Region</Text>
                      <Text style={[styles.infoValue, { color: theme.colors.text }]}>{fieldWorker.region}</Text>
                    </View>
                  </View>

                  <View style={styles.infoRow}>
                    <MaterialCommunityIcons name="phone" size={20} color={theme.colors.primary} />
                    <View style={styles.infoContent}>
                      <Text style={[styles.infoLabel, { color: theme.colors.textSecondary }]}>Phone</Text>
                      <Text style={[styles.infoValue, { color: theme.colors.text }]}>
                        {fieldWorker.profile?.phone || 'Not provided'}
                      </Text>
                    </View>
                  </View>

                  <View style={styles.infoRow}>
                    <MaterialCommunityIcons name="email" size={20} color={theme.colors.primary} />
                    <View style={styles.infoContent}>
                      <Text style={[styles.infoLabel, { color: theme.colors.textSecondary }]}>Email</Text>
                      <Text style={[styles.infoValue, { color: theme.colors.text }]}>{fieldWorker.email}</Text>
                    </View>
                  </View>
                </>
              )}
            </View>
          </ModernCard>
        </Animatable.View>

        {/* Quick Actions Card */}
        <Animatable.View animation="fadeInUp" delay={500}>
          <ModernCard style={styles.card}>
            <View style={styles.cardTitleSection}>
              <MaterialCommunityIcons 
                name="lightning-bolt" 
                size={24} 
                color={theme.colors.primary} 
              />
              <Title style={[styles.cardTitle, { color: theme.colors.text }]}>
                Quick Actions
              </Title>
            </View>
            
            <Divider style={styles.divider} />

            <View style={styles.actionsGrid}>
              <TouchableOpacity 
                style={[styles.actionItem, { backgroundColor: theme.colors.surface }]}
                onPress={() => navigation.navigate('Reports')}
              >
                <Surface style={[styles.actionIcon, { backgroundColor: theme.colors.primary + '20' }]}>
                  <MaterialCommunityIcons name="file-document" size={24} color={theme.colors.primary} />
                </Surface>
                <Text style={[styles.actionText, { color: theme.colors.text }]}>My Reports</Text>
              </TouchableOpacity>

              <TouchableOpacity 
                style={[styles.actionItem, { backgroundColor: theme.colors.surface }]}
                onPress={() => navigation.navigate('Tasks')}
              >
                <Surface style={[styles.actionIcon, { backgroundColor: theme.colors.warning + '20' }]}>
                  <MaterialCommunityIcons name="clipboard-list" size={24} color={theme.colors.warning} />
                </Surface>
                <Text style={[styles.actionText, { color: theme.colors.text }]}>Tasks</Text>
              </TouchableOpacity>

              <TouchableOpacity 
                style={[styles.actionItem, { backgroundColor: theme.colors.surface }]}
                onPress={() => navigation.navigate('Camera')}
              >
                <Surface style={[styles.actionIcon, { backgroundColor: theme.colors.success + '20' }]}>
                  <MaterialCommunityIcons name="camera" size={24} color={theme.colors.success} />
                </Surface>
                <Text style={[styles.actionText, { color: theme.colors.text }]}>New Report</Text>
              </TouchableOpacity>

              <TouchableOpacity 
                style={[styles.actionItem, { backgroundColor: theme.colors.surface }]}
                onPress={() => navigation.navigate('Settings')}
              >
                <Surface style={[styles.actionIcon, { backgroundColor: theme.colors.info + '20' }]}>
                  <MaterialCommunityIcons name="cog" size={24} color={theme.colors.info} />
                </Surface>
                <Text style={[styles.actionText, { color: theme.colors.text }]}>Settings</Text>
              </TouchableOpacity>
            </View>
          </ModernCard>
        </Animatable.View>

        {/* Logout Section */}
        <Animatable.View animation="fadeInUp" delay={600}>
          <View style={styles.logoutSection}>
            <Button
              mode="outlined"
              onPress={handleLogout}
              icon="logout"
              textColor={theme.colors.error}
              style={[styles.logoutButton, { borderColor: theme.colors.error }]}
              contentStyle={styles.buttonContent}
            >
              Logout
            </Button>
          </View>
        </Animatable.View>

        <View style={styles.bottomSpacing} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
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
    marginBottom: 15,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  name: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  email: {
    fontSize: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  workerIdChip: {
    marginTop: 8,
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingTop: 8,
  },
  card: {
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 0,
  },
  cardTitleSection: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 12,
  },
  editButton: {
    borderRadius: 12,
  },
  divider: {
    marginHorizontal: 20,
    marginVertical: 16,
  },
  infoSection: {
    padding: 20,
    paddingTop: 0,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  infoContent: {
    flex: 1,
    marginLeft: 12,
  },
  infoLabel: {
    fontSize: 14,
    marginBottom: 4,
  },
  infoValue: {
    fontSize: 16,
    fontWeight: '500',
  },
  input: {
    marginBottom: 16,
    backgroundColor: 'transparent',
  },
  buttonContainer: {
    marginTop: 8,
  },
  saveButton: {
    borderRadius: 12,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 0,
  },
  actionItem: {
    width: '48%',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    elevation: 1,
  },
  actionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  actionText: {
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
  logoutSection: {
    marginTop: 16,
    paddingHorizontal: 16,
  },
  logoutButton: {
    borderRadius: 12,
    borderWidth: 2,
  },
  bottomSpacing: {
    height: 32,
  },
});

export default ProfileScreen;
