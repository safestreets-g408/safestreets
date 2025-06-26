import React, { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  Image, 
  KeyboardAvoidingView, 
  Platform,
  ScrollView,
  Alert,
  StatusBar
} from 'react-native';
import { TextInput, Title, Paragraph, useTheme } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { loginFieldWorker } from '../utils/auth';
import { useAuth } from '../context/AuthContext';
import { API_BASE_URL } from '../config';
import { GradientButton } from '../components/ui';

const LoginScreen = ({ navigation, route }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [secureTextEntry, setSecureTextEntry] = useState(true);
  const { login } = useAuth();
  const theme = useTheme();

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please enter both email and password');
      return;
    }

    setIsLoading(true);
    
    try {
      console.log(`Attempting login with email: ${email}`);
      console.log('API URL being used:', API_BASE_URL);
      
      const response = await loginFieldWorker(email, password);
      console.log('Login response received:', response ? 'Success' : 'No data');
      
      if (response && response.fieldWorker && response.token) {
        console.log('Login successful, redirecting to MainTabs');
        login(response.fieldWorker);
        navigation.replace('MainTabs');
      } else {
        console.error('Invalid response structure:', response);
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.log('Login error details:', error);
      
      // Handle specific error types
      if (error.name === 'AbortError' || error.message?.includes('timeout') || error.message?.includes('network request')) {
        Alert.alert(
          'Connection Error', 
          'Unable to connect to the server. Please check your internet connection or try again later.',
          [
            {
              text: 'OK',
              onPress: () => console.log('OK Pressed')
            },
            {
              text: 'Debug Info',
              onPress: () => console.log('Current API URL:', API_BASE_URL)
            },
            {
              text: 'Server Details',
              onPress: () => Alert.alert('Server Information', `Trying to connect to: ${API_BASE_URL}\n\nIf you're using the iOS simulator, make sure the backend server is running on your local machine. If using a physical device, ensure you're using your computer's local IP address in config.js.`)
            }
          ]
        );
      } else if (error.message?.includes('Invalid credentials')) {
        Alert.alert(
          'Login Failed', 
          'The email or password you entered is incorrect. Please try again.',
          [
            {
              text: 'OK',
              onPress: () => console.log('OK Pressed')
            },
            {
              text: 'Debug Info',
              onPress: () => Alert.alert('Debug Information', 
                `Email used: ${email}\nPassword length: ${password.length}\nAPI URL: ${API_BASE_URL}`)
            }
          ]
        );
      } else {
        Alert.alert(
          'Login Failed', 
          error.message || 'An unexpected error occurred',
          [
            {
              text: 'OK',
              onPress: () => console.log('OK Pressed')
            },
            {
              text: 'Debug Info',
              onPress: () => Alert.alert('Debug Information', 
                `Error: ${error.message}\nAPI URL: ${API_BASE_URL}`)
            }
          ]
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  const toggleSecureEntry = () => {
    setSecureTextEntry(!secureTextEntry);
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={theme.colors.primaryDark} />
      
      {/* Background Gradient */}
      <LinearGradient
        colors={[theme.colors.primary, theme.colors.primaryDark]}
        style={styles.headerGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
      >
        <Animatable.View animation="fadeIn" duration={1000}>
          <Image 
            source={require('../assets/icon.png')} 
            style={styles.headerLogo}
            resizeMode="contain"
          />
        </Animatable.View>
      </LinearGradient>
      
      <Animatable.View 
        style={styles.formContainer}
        animation="fadeInUpBig"
        duration={800}
      >
        <View style={styles.titleSection}>
          <Title style={[styles.title, { color: theme.colors.text }]}>
            Welcome Back
          </Title>
          <Paragraph style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
            Sign in to continue to SafeStreets
          </Paragraph>
        </View>
        
        <KeyboardAvoidingView 
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.formWrapper}
        >
          <ScrollView 
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            <TextInput
              label="Email Address"
              value={email}
              onChangeText={setEmail}
              style={styles.input}
              mode="outlined"
              autoCapitalize="none"
              keyboardType="email-address"
              left={<TextInput.Icon icon="email" color={theme.colors.primary} />}
              activeOutlineColor={theme.colors.primary}
              outlineColor={theme.colors.border}
              theme={{ roundness: theme.roundness }}
            />

            <TextInput
              label="Password"
              value={password}
              onChangeText={setPassword}
              style={styles.input}
              mode="outlined"
              secureTextEntry={secureTextEntry}
              activeOutlineColor={theme.colors.primary}
              outlineColor={theme.colors.border}
              theme={{ roundness: theme.roundness }}
              left={<TextInput.Icon icon="lock" color={theme.colors.primary} />}
              right={
                <TextInput.Icon 
                  icon={secureTextEntry ? "eye" : "eye-off"} 
                  onPress={toggleSecureEntry} 
                  color={theme.colors.secondary}
                />
              }
            />
            
            <View style={styles.forgotPasswordContainer}>
              <TouchableOpacity 
                onPress={() => Alert.alert('Reset Password', 'A password reset link will be sent to your email address.')}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Text style={[styles.forgotPasswordText, { color: theme.colors.primary }]}>
                  Forgot Password?
                </Text>
              </TouchableOpacity>
            </View>

            <GradientButton
              title="SIGN IN"
              onPress={handleLogin}
              loading={isLoading}
              disabled={isLoading}
              mode="primary"
              style={styles.loginButton}
              icon={<MaterialCommunityIcons name="login" size={20} color="white" style={{ marginRight: 8 }} />}
            />
            
            <View style={styles.dividerContainer}>
              <View style={[styles.divider, { backgroundColor: theme.colors.border }]} />
              <Text style={[styles.dividerText, { color: theme.colors.textSecondary }]}>
                Field Worker Portal
              </Text>
              <View style={[styles.divider, { backgroundColor: theme.colors.border }]} />
            </View>
            
            <Paragraph style={[styles.supportText, { color: theme.colors.textSecondary }]}>
              Having trouble logging in? Contact support at support@safestreets.com
            </Paragraph>
          </ScrollView>
        </KeyboardAvoidingView>
      </Animatable.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
  headerGradient: {
    height: '35%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerLogo: {
    width: 120,
    height: 120,
    tintColor: 'white',
  },
  formContainer: {
    flex: 1,
    backgroundColor: 'white',
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    marginTop: -40,
    paddingHorizontal: 24,
    paddingTop: 30,
  },
  titleSection: {
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 16,
    marginTop: 8,
  },
  formWrapper: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: 24,
  },
  input: {
    marginBottom: 18,
    backgroundColor: 'white',
    height: 56,
  },
  loginButton: {
    marginTop: 16,
    height: 50,
  },
  forgotPasswordContainer: {
    alignItems: 'flex-end',
    marginTop: -8,
    marginBottom: 16,
  },
  forgotPasswordText: {
    fontWeight: '500',
    fontSize: 14,
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 28,
  },
  divider: {
    flex: 1,
    height: 1,
  },
  dividerText: {
    paddingHorizontal: 16,
    fontSize: 14,
  },
  supportText: {
    textAlign: 'center',
    fontSize: 13,
    marginTop: 8,
  }
});

export default LoginScreen;
