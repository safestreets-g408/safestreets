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

module.exports = {
  sendTenantCredentials
};
