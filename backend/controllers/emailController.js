const { sendMail } = require('../utils/emailService');

// Send tenant credentials via email
const sendTenantCredentials = async (req, res) => {
  try {
    const { email, tenantName, adminName, password, loginUrl } = req.body;

    if (!email || !tenantName || !adminName || !password || !loginUrl) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    // Create email content
    const subject = `SafeStreets Platform - Your Tenant Account Details`;
    
    const html = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #1976d2; padding: 20px; text-align: center; color: white;">
          <h1 style="margin: 0;">SafeStreets Platform</h1>
        </div>
        
        <div style="padding: 20px; background-color: #f8f8f8; border: 1px solid #ddd;">
          <h2>Welcome to SafeStreets!</h2>
          
          <p>Dear ${adminName},</p>
          
          <p>Your access request has been approved and a tenant account has been created for <strong>${tenantName}</strong>.</p>
          
          <p>Please use the following credentials to log in to the SafeStreets platform:</p>
          
          <div style="background-color: #fff; padding: 15px; border-left: 4px solid #1976d2; margin: 20px 0;">
            <p><strong>Tenant:</strong> ${tenantName}</p>
            <p><strong>Email:</strong> ${email}</p>
            <p><strong>Password:</strong> ${password}</p>
            <p><strong>Login URL:</strong> <a href="${loginUrl}">${loginUrl}</a></p>
          </div>
          
          <p>For security reasons, we recommend changing your password after your first login.</p>
          
          <p>If you have any questions or need assistance, please contact our support team.</p>
          
          <p>Best regards,<br>SafeStreets Team</p>
        </div>
        
        <div style="text-align: center; color: #666; padding: 10px; font-size: 12px;">
          <p>This is an automated message. Please do not reply to this email.</p>
        </div>
      </div>
    `;

    // Send the email
    await sendMail({
      to: email,
      subject,
      html,
      text: `
Welcome to SafeStreets!

Dear ${adminName},

Your access request has been approved and a tenant account has been created for ${tenantName}.

Please use the following credentials to log in to the SafeStreets platform:

Tenant: ${tenantName}
Email: ${email}
Password: ${password}
Login URL: ${loginUrl}

For security reasons, we recommend changing your password after your first login.

If you have any questions or need assistance, please contact our support team.

Best regards,
SafeStreets Team

This is an automated message. Please do not reply to this email.
      `
    });

    res.status(200).json({ success: true, message: 'Credentials sent successfully' });
  } catch (error) {
    console.error('Error sending tenant credentials:', error);
    res.status(500).json({ message: 'Failed to send credentials email', error: error.message });
  }
};

// Handle newsletter subscription
const subscribeToNewsletter = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ message: 'Email is required' });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ message: 'Invalid email format' });
    }

    // Send confirmation email
    const subject = `SafeStreets - Newsletter Subscription Confirmation`;
    
    const html = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #1976d2; padding: 20px; text-align: center; color: white;">
          <h1 style="margin: 0;">SafeStreets</h1>
        </div>
        
        <div style="padding: 20px; background-color: #f8f8f8; border: 1px solid #ddd;">
          <h2>Thank You for Subscribing!</h2>
          
          <p>You've successfully subscribed to the SafeStreets newsletter.</p>
          
          <p>You'll now receive regular updates about our product features, AI advancements, 
          and best practices in road maintenance.</p>
          
          <p>If you didn't subscribe to our newsletter, please disregard this email.</p>
          
          <p>Best regards,<br>SafeStreets Team</p>
        </div>
        
        <div style="text-align: center; color: #666; padding: 10px; font-size: 12px;">
          <p>© ${new Date().getFullYear()} SafeStreets AI. All rights reserved.</p>
          <p>You can unsubscribe at any time by clicking the unsubscribe link in our emails.</p>
        </div>
      </div>
    `;

    // Send the confirmation email
    await sendMail({
      to: email,
      subject,
      html,
      text: `
Thank You for Subscribing!

You've successfully subscribed to the SafeStreets newsletter.

You'll now receive regular updates about our product features, AI advancements, 
and best practices in road maintenance.

If you didn't subscribe to our newsletter, please disregard this email.

Best regards,
SafeStreets Team

© ${new Date().getFullYear()} SafeStreets AI. All rights reserved.
You can unsubscribe at any time by clicking the unsubscribe link in our emails.
      `
    });

    // In a real application, you would save the email to a subscribers database
    // For this example, we'll just send a success response

    res.status(200).json({ 
      success: true, 
      message: 'Subscription successful! You will now receive our newsletter.' 
    });
  } catch (error) {
    console.error('Error subscribing to newsletter:', error);
    res.status(500).json({ 
      message: 'Failed to process subscription', 
      error: error.message 
    });
  }
};

module.exports = {
  sendTenantCredentials,
  subscribeToNewsletter
};
