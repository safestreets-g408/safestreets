const nodemailer = require('nodemailer');

// This is a simple email service. In production, you might want to use
// a service like SendGrid, Mailgun, or AWS SES.
const sendMail = async ({ to, subject, text, html }) => {
  try {
    // Create a test account if no SMTP credentials are provided
    // This is useful for development
    let testAccount, transporter;

    if (!process.env.SMTP_HOST) {
      testAccount = await nodemailer.createTestAccount();
      
      transporter = nodemailer.createTransport({
        host: 'smtp.ethereal.email',
        port: 587,
        secure: false, // true for 465, false for other ports
        auth: {
          user: testAccount.user,
          pass: testAccount.pass,
        },
      });
      
      console.log('Using ethereal test email account');
    } else {
      // Use configured SMTP server
      transporter = nodemailer.createTransport({
        host: process.env.SMTP_HOST,
        port: process.env.SMTP_PORT || 587,
        secure: process.env.SMTP_SECURE === 'true',
        auth: {
          user: process.env.SMTP_USER,
          pass: process.env.SMTP_PASS,
        },
      });
      
      console.log('Using configured SMTP server');
    }

    // Send mail with defined transport object
    const info = await transporter.sendMail({
      from: process.env.SMTP_FROM || '"SafeStreets" <noreply@safestreets.com>',
      to: Array.isArray(to) ? to.join(', ') : to,
      subject,
      text,
      html: html || text.replace(/\n/g, '<br>'),
    });

    console.log('Message sent: %s', info.messageId);
    
    // For test accounts, log preview URL
    if (testAccount) {
      console.log('Preview URL: %s', nodemailer.getTestMessageUrl(info));
    }

    return info;
  } catch (error) {
    console.error('Error sending email:', error);
    throw error;
  }
};

module.exports = { sendMail };
