const axios = require('axios');

/**
 * Utility service for sending notifications
 */
class NotificationService {
  /**
   * Send a push notification to a specific device token
   * @param {string} token - The Expo push token
   * @param {string} title - Notification title
   * @param {string} body - Notification body text
   * @param {Object} data - Additional data payload
   * @returns {Promise<boolean>} Success status
   */
  static async sendPushNotification(token, title, body, data = {}) {
    try {
      // Check if token is valid
      if (!token || !token.includes('ExponentPushToken')) {
        console.warn('Invalid Expo push token format:', token);
        return false;
      }

      // Prepare notification message
      const message = {
        to: token,
        sound: 'default',
        title,
        body,
        data: {
          ...data,
          timestamp: new Date().toISOString(),
        },
        badge: 1,
      };

      // Send to Expo push notification service
      const response = await axios.post(
        'https://exp.host/--/api/v2/push/send',
        message,
        {
          headers: {
            Accept: 'application/json',
            'Accept-encoding': 'gzip, deflate',
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.data.data && response.data.data.status === 'ok') {
        return true;
      } else {
        console.warn('Expo push notification response:', response.data);
        return false;
      }
    } catch (error) {
      console.error('Error sending push notification:', error);
      return false;
    }
  }

  /**
   * Send a notification to multiple tokens
   * @param {string[]} tokens - Array of Expo push tokens
   * @param {string} title - Notification title
   * @param {string} body - Notification body text
   * @param {Object} data - Additional data payload
   * @returns {Promise<boolean>} Success status
   */
  static async sendPushNotifications(tokens, title, body, data = {}) {
    if (!tokens || !tokens.length) {
      return false;
    }

    try {
      // Prepare messages
      const messages = tokens.map(token => ({
        to: token,
        sound: 'default',
        title,
        body,
        data: {
          ...data,
          timestamp: new Date().toISOString(),
        },
        badge: 1,
      }));

      // Send batch notification
      const response = await axios.post(
        'https://exp.host/--/api/v2/push/send',
        messages,
        {
          headers: {
            Accept: 'application/json',
            'Accept-encoding': 'gzip, deflate',
            'Content-Type': 'application/json',
          },
        }
      );

      return true;
    } catch (error) {
      console.error('Error sending batch push notifications:', error);
      return false;
    }
  }

  /**
   * Create a notification payload
   * @param {string} title - Notification title
   * @param {string} body - Notification body text
   * @param {string} type - Notification type (task, report, alert, message)
   * @param {Object} data - Additional data
   * @returns {Object} Notification payload
   */
  static createNotificationPayload(title, body, type = 'system', data = {}) {
    return {
      title,
      body,
      data: {
        type,
        ...data,
        timestamp: new Date().toISOString(),
      },
    };
  }
}

module.exports = NotificationService;
