# Mobile Application Guide

This document provides detailed information about the SafeStreets mobile application designed for field workers to capture, submit, and track road damage reports.

## Overview

The SafeStreets mobile application is built with React Native and Expo, offering a cross-platform solution for field workers to capture road damage images, submit reports, and track their status. The app features a user-friendly interface with advanced camera integration, GPS auto-tagging, and offline synchronization capabilities.

## Installation

### Field Worker Device Setup

1. **Install Expo Go**:
   - Download from [App Store](https://apps.apple.com/app/expo-go/id982107779) (iOS)
   - Download from [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent) (Android)

2. **Access the App**:
   - Scan the QR code provided by your administrator
   - OR use the published Expo link (if available)

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/safestreets-g408/safestreets.git
   cd safestreets/apps/user-app
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npx expo start
   ```

4. **Connect to the app**:
   - Scan the QR code with your device
   - Run on iOS simulator: `npx expo start --ios`
   - Run on Android emulator: `npx expo start --android`

## Authentication

### Login Process

1. Open the SafeStreets mobile app
2. Enter your credentials:
   - **Email**: `firstname.lastname@safestreets.worker`
   - **Password**: First 3 letters of your first name + your worker ID
   - Example: For John Doe with worker ID FW001, the credentials would be:
     - Email: `john.doe@safestreets.worker`
     - Password: `johFW001`

### Profile Management

After logging in, you can access your profile from the Profile tab:
- View personal information
- Update contact details
- Set notification preferences
- View work history and statistics
- Log out

## Main Functionality

### Capturing Road Damage

1. **Access the Camera**:
   - Navigate to the Camera tab
   - Allow camera and location permissions when prompted

2. **Capture an Image**:
   - Position the camera to clearly show the road damage
   - Tap the capture button
   - Review the captured image
   - Retake if necessary

3. **Add Details**:
   - Enter a description (optional)
   - Select damage severity (if prompted)
   - Add specific location details (if GPS is inaccurate)

4. **Submit the Report**:
   - Tap the Submit button
   - The app will:
     - Embed GPS coordinates automatically
     - Upload the image to the server
     - Queue for AI analysis
   - If offline, the report will be saved locally and synchronized when connectivity is restored

### Tracking Reports

The Reports tab allows you to:
- View a list of all submitted reports
- Filter by status (Pending, Processing, Analyzed, Assigned, Completed)
- Sort by date, severity, or location
- View detailed information for each report, including:
  - Status
  - AI analysis results (classification and confidence)
  - Location on map
  - Timestamp

### Task Management

The Tasks tab displays:
- Assigned repair tasks
- Task details:
  - Location
  - Damage type and severity
  - Priority level
  - Due date (if applicable)
- Actions:
  - Update status (In Progress, Completed)
  - Add notes
  - Upload completion photos

## Features

### Advanced Camera Integration

- High-quality image capture using Expo Camera
- Automatic focus and exposure adjustment
- Image preview and retake capabilities
- Flash control for low-light conditions

### GPS Auto-Tagging

- Automatic embedding of GPS coordinates
- Location accuracy indicator
- Manual location adjustment if needed
- Geocoding to show address when available

### Offline Synchronization

- AsyncStorage for local data persistence
- Automatic synchronization when connectivity is restored
- Background uploads
- Conflict resolution for parallel updates

### Status Tracking

- Real-time status updates via push notifications
- Visual indicators for each report status
- Timeline view of status changes
- Detailed AI analysis results when available

### User Interface

- Material Design with React Native Paper
- Dark/light mode support
- Responsive layout for various device sizes
- Gesture-based navigation
- Animated transitions and feedback

## Troubleshooting

### Common Issues

#### Camera Not Working
- Ensure camera permissions are granted
- Check if another app is using the camera
- Restart the app
- Update Expo Go to the latest version

#### Location Not Accurate
- Ensure location services are enabled
- Move to an open area for better GPS signal
- Wait for a few seconds to get a more accurate reading
- Try manually adjusting the location

#### Upload Failures
- Check internet connectivity
- Ensure the file size isn't too large
- Try again on a different network
- Verify that your authentication hasn't expired

#### App Crashes
- Update to the latest version
- Clear app cache
- Restart your device
- Contact support if the issue persists

### Support Contact

If you encounter persistent issues:
- Email: support@safestreets.com
- In-app feedback form (Settings > Send Feedback)
- Contact your administrator

## Privacy and Data Usage

The SafeStreets mobile app collects:
- Road damage images
- GPS location data
- Device information for diagnostics
- Usage statistics for app improvement

All data is securely transmitted and stored according to the organization's privacy policy. Location data is only collected when capturing road damage images and is not tracked continuously.

## Update Process

The app will prompt you when updates are available:
- Minor updates can be installed directly through Expo Go
- Major updates may require downloading a new version

Always keep the app updated to ensure access to the latest features and security improvements.

## Keyboard Shortcuts (for Testing on Simulators)

When using the app on iOS or Android simulators, you can use these keyboard shortcuts:
- Capture Photo: `Enter` or `Space`
- Toggle Camera: `T`
- Toggle Flash: `F`
- Cancel: `Escape`
