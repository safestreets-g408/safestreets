# Road Damage Detection and Management System

This document outlines the architecture and features of a road damage detection and management system, designed to streamline maintenance operations.

## 1. Mobile Application (for Maintenance Teams)

* **Image Capture:**
    * Field workers can capture images of road damages using their mobile devices.
* **GPS Tagging:**
    * Automatic GPS location tagging is embedded within each captured image.
* **Damage Submission:**
    * Images and GPS coordinates are securely transmitted to the backend server via an API.
* **Status Tracking:**
    * Users can view the history of their submitted reports, including their current status (pending/resolved).
* **Offline Support (Optional):**
    * The application can store images offline and synchronize them with the server when an internet connection is available.

## 2. AI-Powered Damage Detection (Vision Transformer Model)

* **Image Classification:**
    * The system employs a Vision Transformer (ViT) model to classify road damages into the following categories:
        * Potholes
        * Surface cracks
        * Alligator cracking
        * Erosion/wear
* **Severity Assessment:**
    * The model assesses the severity of the damage based on factors such as:
        * Crack width
        * Area size
        * Damage density
* **Text Summary Generation:**
    * The system generates a concise text summary of the damage, including:
        * Type: (e.g., Pothole)
        * Severity: (e.g., High)
        * Priority: (e.g., Urgent)
        * Action: (e.g., Immediate repair suggested)
* **Performance Optimization:**
    * Preprocessing techniques are used to enhance image quality for better ViT performance.
    * Configurable confidence thresholds are implemented.

## 3. Backend (Node.js + Express + MongoDB)

* **Image API:**
    * Handles the reception, storage, and processing of uploaded images.
* **ViT API Trigger:**
    * Sends images to the machine learning model and parses the response.
* **Database Management:**
    * Manages user data, images, classification results, summaries, and repair history using MongoDB.
* **Email Automation:**
    * Generates and sends detailed reports to assigned administrators or authorities, including:
        * Image
        * Summary
        * Location
* **Task Assignment API:**
    * Admin users can assign tasks to repair teams through the dashboard.

## 4. Web Dashboard (React + MUI)

* **Admin Login Panel:**
    * Secure login for administrative access.
* **Visual Reports:**
    * Displays a list of reported damages with filtering options (date, severity, region, damage type).
* **Map View (Mapbox / Leaflet):**
    * Presents a location-based heatmap of reported damages.
* **Analytics & Insights:**
    * Provides daily/weekly trends, severity distribution, and identifies most-affected zones.
* **Repair Management:**
    * Allows administrators to assign tasks to field workers and update the status of repairs (pending, in-progress, resolved).
* **Historical Analysis:**
    * Enables viewing of old damage reports, comparing before/after photos, and analyzing repair frequency by location.

## 5. Field Worker Management

* **Field Worker Profiles:**
    * Comprehensive profile management for field workers
    * Work details: name, worker ID, specialization, region
    * Contact details: work email, personal email, phone number
    * Performance metrics: active assignments, completed reports

* **Assignment Management:**
    * Track active assignments per worker
    * View detailed assignment information
    * Enforce maximum assignment limits to prevent overloading
    * Geographic assignment matching based on worker's region

* **Daily Updates:**
    * Automated daily email updates to field workers
    * Modern, responsive HTML email templates
    * Personalized assignment summaries
    * Status tracking and prioritization

## 6. Notifications & Alerts

* **Email Notifications:**
    * Daily status updates to field workers via personal email
    * Sends email alerts to administrators for new high-priority damage reports
    * Configurable notification preferences
    * Responsive HTML templates with status color-coding

* **Push Notifications:**
    * Real-time alerts to field teams for new assignments
    * Status update notifications
    * Priority-based notification system

* **Task Reminders:**
    * Sends reminders for tasks that are pending beyond a specified deadline
    * Escalation alerts for overdue high-priority tasks

## 7. Security & Access Control

* **JWT Authentication:**
    * Uses JSON Web Tokens (JWT) for secure authentication of users and administrators.
* **Role-Based Access:**
    * Defines different access levels based on user roles:
        * Maintenance Team: Upload images only.
        * Admin: Full dashboard access, task assignment, report viewing.
* **Data Validation & Sanitization:**
    * Implements data validation and sanitization techniques for uploaded data to prevent security vulnerabilities.