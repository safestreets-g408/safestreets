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
  Alert
} from 'react-native';
import { TextInput, Button, Title, Paragraph } from 'react-native-paper';
import { loginFieldWorker } from '../utils/auth';
import { useAuth } from '../context/AuthContext';
import { API_BASE_URL } from '../config';

const LoginScreen = ({ navigation, route }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [secureTextEntry, setSecureTextEntry] = useState(true);
  const { login } = useAuth();

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
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.logoContainer}>
          <Image 
            source={require('../assets/icon.png')} 
            style={styles.logo}
            resizeMode="contain"
          />
          <Title style={styles.appTitle}>Safe Streets</Title>
          <Paragraph style={styles.appSubtitle}>
            Official road maintenance reporting system
          </Paragraph>
        </View>

        <View style={styles.formContainer}>
          <TextInput
            label="Email"
            value={email}
            onChangeText={setEmail}
            style={styles.input}
            mode="outlined"
            autoCapitalize="none"
            keyboardType="email-address"
            left={<TextInput.Icon name="email" color="#003366" />}
            outlineColor="#E0E6ED"
            activeOutlineColor="#003366"
            theme={{ colors: { primary: '#003366', text: '#263238' } }}
          />

          <TextInput
            label="Password"
            value={password}
            onChangeText={setPassword}
            style={styles.input}
            mode="outlined"
            secureTextEntry={secureTextEntry}
            outlineColor="#E0E6ED"
            activeOutlineColor="#003366"
            theme={{ colors: { primary: '#003366', text: '#263238' } }}
            right={
              <TextInput.Icon 
                name={secureTextEntry ? "eye" : "eye-off"} 
                onPress={toggleSecureEntry} 
                color="#003366"
              />
            }
            left={<TextInput.Icon name="lock" color="#003366" />}
          />

          <Button 
            mode="contained" 
            onPress={handleLogin} 
            style={styles.loginButton}
            buttonColor="#003366"
            loading={isLoading}
            disabled={isLoading}
            icon="login"
          >
            SIGN IN
          </Button>

          <View style={styles.forgotPasswordContainer}>
            <TouchableOpacity onPress={() => Alert.alert('Reset Password', 'A password reset link will be sent to your email address.')}>
              <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f9fc',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logo: {
    width: 120,
    height: 120,
    marginBottom: 16,
  },
  appTitle: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#003366',
    letterSpacing: 0.5,
  },
  appSubtitle: {
    fontSize: 15,
    color: '#546E7A',
    textAlign: 'center',
    marginTop: 8,
    letterSpacing: 0.2,
    maxWidth: 280,
  },
  formContainer: {
    width: '100%',
  },
  input: {
    marginBottom: 18,
    backgroundColor: 'white',
    borderRadius: 4,
    height: 56,
  },
  loginButton: {
    marginTop: 10,
    borderRadius: 4,
    height: 48,
    justifyContent: 'center',
    elevation: 1,
  },
  forgotPasswordContainer: {
    alignItems: 'center',
    marginTop: 16,
  },
  forgotPasswordText: {
    color: '#0055a4',
    fontWeight: '500',
    fontSize: 14,
    letterSpacing: 0.2,
  },
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 40,
    marginBottom: 20,
  },
  signupText: {
    color: '#455A64',
    fontSize: 14,
  },
  signupLink: {
    color: '#003366',
    fontWeight: 'bold',
    fontSize: 14,
  },
});

export default LoginScreen;
