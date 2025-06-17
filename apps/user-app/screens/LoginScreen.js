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
      const response = await loginFieldWorker(email, password);
      login(response.fieldWorker);
      navigation.replace('MainTabs');
    } catch (error) {
      Alert.alert('Login Failed', error.message || 'Invalid credentials');
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

          {/* Field Worker Credentials Info */}
          <View style={styles.credentialsInfo}>
            <Text style={styles.credentialsTitle}>Field Worker Credentials</Text>
            <Text style={styles.credentialsText}>
              • Email: firstname.lastname@safestreets.worker
            </Text>
            <Text style={styles.credentialsText}>
              • Password: first3lettersofname + workerID
            </Text>
            <Text style={styles.credentialsExample}>
              Example: john.doe@safestreets.worker, password: johFW001
            </Text>
          </View>

          <View style={styles.forgotPasswordContainer}>
            <TouchableOpacity onPress={() => Alert.alert('Reset Password', 'A password reset link will be sent to your email address.')}>
              <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>OR</Text>
            <View style={styles.dividerLine} />
          </View>

          <Button 
            mode="outlined" 
            onPress={() => Alert.alert('Security Access', 'Government ID verification required for official access.')} 
            style={styles.govLoginButton}
            icon="shield-account"
            textColor="#003366"
          >
            OFFICIAL GOVERNMENT LOGIN
          </Button>
        </View>

        <View style={styles.signupContainer}>
          <Text style={styles.signupText}>First time user? </Text>
          <TouchableOpacity onPress={() => Alert.alert('Account Creation', 'Please contact your local authority to create an official account.')}>
            <Text style={styles.signupLink}>Create Account</Text>
          </TouchableOpacity>
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
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#E0E6ED',
  },
  dividerText: {
    marginHorizontal: 16,
    color: '#78909C',
    fontSize: 14,
    fontWeight: '500',
  },
  govLoginButton: {
    borderColor: '#003366',
    borderWidth: 1.5,
    marginTop: 8,
    height: 48,
    justifyContent: 'center',
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
  credentialsInfo: {
    backgroundColor: '#E3F2FD',
    padding: 16,
    borderRadius: 8,
    marginTop: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#003366',
  },
  credentialsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#003366',
    marginBottom: 8,
  },
  credentialsText: {
    fontSize: 14,
    color: '#37474F',
    marginBottom: 4,
    lineHeight: 20,
  },
  credentialsExample: {
    fontSize: 12,
    color: '#546E7A',
    marginTop: 8,
    fontStyle: 'italic',
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
