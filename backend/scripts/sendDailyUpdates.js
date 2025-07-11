/**
 * Script to send daily updates to all field workers across all tenants
 * This script can be run as a cron job to send daily updates at a specific time
 * Example cron schedule: 0 7 * * * cd /path/to/safestreets && node backend/scripts/sendDailyUpdates.js >> /path/to/logs/daily-updates.log 2>&1
 * (This will run the script every day at 7:00 AM and log output to daily-updates.log)
 */

require('dotenv').config();
const mongoose = require('mongoose');
const FieldWorker = require('../models/FieldWorker');
const DamageReport = require('../models/DamageReport');
const { sendMail } = require('../utils/emailService');

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected for daily updates'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

// Function to send daily updates to a single field worker
async function sendDailyUpdateToWorker(worker) {
  try {
    // Skip if no personal email or not opted in
    if (!worker.profile?.personalEmail || worker.profile?.receiveDailyUpdates === false) {
      return {
        workerId: worker.workerId,
        name: worker.name,
        status: 'Skipped',
        reason: 'No personal email or opted out of daily updates'
      };
    }

    // Get worker's assigned reports
    const assignedReports = await DamageReport.find({
      assignedTo: worker._id,
      status: { $nin: ['Completed', 'Closed'] }
    }).select('reportId location damageType severity status createdAt');

    // Format the email content
    const today = new Date();
    const dateStr = today.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });

    const emailSubject = `SafeStreets Daily Update - ${dateStr}`;
    
    // Create a modern HTML email template
    let htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>SafeStreets Daily Update</title>
      <style>
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          line-height: 1.6;
          color: #333;
          margin: 0;
          padding: 0;
        }
        .container {
          max-width: 600px;
          margin: 0 auto;
          padding: 20px;
        }
        .header {
          background-color: #2563eb;
          color: white;
          padding: 20px;
          text-align: center;
          border-radius: 5px 5px 0 0;
        }
        .content {
          background-color: #f9fafb;
          padding: 20px;
          border-radius: 0 0 5px 5px;
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .footer {
          text-align: center;
          margin-top: 20px;
          font-size: 12px;
          color: #6b7280;
        }
        h1, h2 {
          margin: 0;
          padding: 0;
        }
        h1 {
          font-size: 24px;
          margin-bottom: 10px;
        }
        h2 {
          font-size: 20px;
          margin-bottom: 15px;
          color: #2563eb;
        }
        .date {
          font-size: 16px;
          color: #f8fafc;
        }
        .assignments {
          margin-top: 20px;
        }
        .report-card {
          background: white;
          border-radius: 5px;
          padding: 15px;
          margin-bottom: 15px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          border-left: 4px solid #2563eb;
        }
        .report-header {
          display: flex;
          justify-content: space-between;
          border-bottom: 1px solid #e5e7eb;
          padding-bottom: 10px;
          margin-bottom: 10px;
        }
        .report-title {
          font-weight: bold;
          font-size: 16px;
          color: #1e3a8a;
        }
        .report-status {
          padding: 3px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }
        .status-pending {
          background-color: #fef9c3;
          color: #854d0e;
        }
        .status-in-progress {
          background-color: #dbeafe;
          color: #1e40af;
        }
        .status-review {
          background-color: #fae8ff;
          color: #86198f;
        }
        .report-detail {
          margin: 5px 0;
          display: flex;
        }
        .detail-label {
          font-weight: 500;
          width: 90px;
          color: #4b5563;
        }
        .severity-high {
          color: #b91c1c;
          font-weight: 500;
        }
        .severity-medium {
          color: #b45309;
          font-weight: 500;
        }
        .severity-low {
          color: #047857;
          font-weight: 500;
        }
        .cta-button {
          display: block;
          background-color: #2563eb;
          color: white;
          text-align: center;
          padding: 12px 20px;
          text-decoration: none;
          border-radius: 5px;
          margin: 25px auto 15px;
          width: 200px;
          font-weight: 500;
        }
        .no-assignments {
          text-align: center;
          padding: 30px 0;
          color: #6b7280;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>SafeStreets Daily Update</h1>
          <div class="date">${dateStr}</div>
        </div>
        <div class="content">
          <p>Hello ${worker.name},</p>
          <p>Here's your daily summary of assignments and tasks.</p>
          
          <div class="assignments">
            <h2>${assignedReports.length > 0 ? 'Your Active Assignments' : 'Assignment Summary'}</h2>`;
    
    if (assignedReports.length === 0) {
      htmlContent += `
            <div class="no-assignments">
              <p>You currently have no active assignments.</p>
              <p>Enjoy your day! We'll notify you when new tasks are assigned.</p>
            </div>`;
    } else {
      htmlContent += `
            <p>You have <strong>${assignedReports.length}</strong> active assignment${assignedReports.length > 1 ? 's' : ''} that require your attention:</p>`;
      
      assignedReports.forEach((report, index) => {
        // Determine status class for styling
        let statusClass = '';
        switch(report.status) {
          case 'Pending':
            statusClass = 'status-pending';
            break;
          case 'In Progress':
            statusClass = 'status-in-progress';
            break;
          case 'Under Review':
            statusClass = 'status-review';
            break;
          default:
            statusClass = 'status-pending';
        }
        
        // Determine severity class
        let severityClass = '';
        switch(report.severity) {
          case 'High':
            severityClass = 'severity-high';
            break;
          case 'Medium':
            severityClass = 'severity-medium';
            break;
          case 'Low':
            severityClass = 'severity-low';
            break;
          default:
            severityClass = '';
        }

        // Format creation date nicely
        const createdDate = new Date(report.createdAt).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        });
        
        htmlContent += `
            <div class="report-card">
              <div class="report-header">
                <span class="report-title">Report #${report.reportId}</span>
                <span class="report-status ${statusClass}">${report.status}</span>
              </div>
              <div class="report-detail">
                <span class="detail-label">Location:</span>
                <span>${report.location.address || 'Unknown'}</span>
              </div>
              <div class="report-detail">
                <span class="detail-label">Damage Type:</span>
                <span>${report.damageType}</span>
              </div>
              <div class="report-detail">
                <span class="detail-label">Severity:</span>
                <span class="${severityClass}">${report.severity}</span>
              </div>
              <div class="report-detail">
                <span class="detail-label">Created:</span>
                <span>${createdDate}</span>
              </div>
            </div>`;
      });
    }
    
    htmlContent += `
          </div>
          
          <a href="#" class="cta-button">View In App</a>
          
          <p>To view more details or update the status of your assignments, please log in to the SafeStreets Field Worker application.</p>
          
          <p>Thank you for your hard work keeping our streets safe!</p>
          <p>Best regards,<br>The SafeStreets Team</p>
        </div>
        <div class="footer">
          <p>© 2025 SafeStreets. All rights reserved.</p>
          <p>This email was sent to you as part of your daily updates subscription.</p>
        </div>
      </div>
    </body>
    </html>`;
    
    // Create a plain text version as fallback
    let plainText = `Hello ${worker.name},\n\n`;
    plainText += `Here's your daily update for ${dateStr}:\n\n`;
    
    if (assignedReports.length === 0) {
      plainText += "You currently have no active assignments.\n\n";
    } else {
      plainText += `You have ${assignedReports.length} active assignment(s):\n\n`;
      
      assignedReports.forEach((report, index) => {
        plainText += `${index + 1}. Report ID: ${report.reportId}\n`;
        plainText += `   Location: ${report.location.address || 'Unknown'}\n`;
        plainText += `   Damage Type: ${report.damageType}\n`;
        plainText += `   Severity: ${report.severity}\n`;
        plainText += `   Status: ${report.status}\n`;
        plainText += `   Created: ${new Date(report.createdAt).toLocaleDateString()}\n\n`;
      });
    }
    
    plainText += "To view more details or update the status of your assignments, please log in to the SafeStreets Field Worker application.\n\n";
    plainText += "Thank you for your hard work keeping our streets safe!\n\n";
    plainText += "Best regards,\nThe SafeStreets Team";
    
    // Send the email with HTML content and plain text fallback
    await sendMail({
      to: worker.profile.personalEmail,
      subject: emailSubject,
      text: plainText,
      html: htmlContent
    });
    
    return {
      workerId: worker.workerId,
      name: worker.name,
      email: worker.profile.personalEmail,
      status: 'Success',
      assignedReportsCount: assignedReports.length
    };
  } catch (error) {
    console.error(`Error sending daily update to ${worker.name}:`, error);
    return {
      workerId: worker.workerId,
      name: worker.name,
      status: 'Failed',
      error: error.message
    };
  }
}

// Main function to send daily updates to all field workers
async function sendDailyUpdates() {
  try {
    console.log('Starting daily updates process...');
    
    // Find all field workers with personal emails who opted for daily updates
    const fieldWorkers = await FieldWorker.find({
      'profile.personalEmail': { $exists: true, $ne: '' },
      'profile.receiveDailyUpdates': { $ne: false }
    }).select('-password');
    
    console.log(`Found ${fieldWorkers.length} field workers with personal emails`);
    
    if (fieldWorkers.length === 0) {
      console.log('No field workers to send updates to.');
      mongoose.disconnect();
      return;
    }
    
    // Send emails in parallel with rate limiting (10 at a time)
    const results = [];
    const batchSize = 10;
    
    for (let i = 0; i < fieldWorkers.length; i += batchSize) {
      const batch = fieldWorkers.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(fieldWorkers.length/batchSize)}`);
      
      const batchResults = await Promise.all(batch.map(sendDailyUpdateToWorker));
      results.push(...batchResults);
      
      // Small delay between batches to prevent email rate limiting
      if (i + batchSize < fieldWorkers.length) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    // Summarize results
    const successCount = results.filter(r => r.status === 'Success').length;
    const failureCount = results.filter(r => r.status === 'Failed').length;
    const skippedCount = results.filter(r => r.status === 'Skipped').length;
    
    console.log('Daily updates completed:');
    console.log(`- Successful: ${successCount}`);
    console.log(`- Failed: ${failureCount}`);
    console.log(`- Skipped: ${skippedCount}`);
    console.log(`- Total processed: ${results.length}`);
    
    if (failureCount > 0) {
      console.log('\nFailed updates:');
      results
        .filter(r => r.status === 'Failed')
        .forEach(r => console.log(`- ${r.name} (${r.workerId}): ${r.error}`));
    }
  } catch (error) {
    console.error('Error in daily updates process:', error);
  } finally {
    mongoose.disconnect();
    console.log('Daily updates process completed.');
  }
}

// Execute the function
sendDailyUpdates();
