const FieldWorker = require('../models/FieldWorker');

const addFieldWorker = async (req, res) => {
    try {
        const { 
            name, 
            workerId, 
            specialization, 
            region, 
            email, 
            password,
            phone,
            personalEmail,
            status
        } = req.body;

        // Check for duplicate email/workerId within the same tenant
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        const existingWorker = await FieldWorker.findOne({ 
            $or: [{ workerId }, { email }],
            ...tenantFilter // Only check within the same tenant
        });
        
        if (existingWorker) {
            return res.status(400).json({ 
                message: 'Worker ID or email already exists' 
            });
        }

        // Create profile object with phone and personalEmail
        const profile = {
            phone: phone || '',
            personalEmail: personalEmail || '',
            isActive: true,
            lastActive: new Date(),
            totalReportsHandled: 0,
            receiveDailyUpdates: personalEmail ? true : false
        };

        const fieldWorker = new FieldWorker({
            name,
            workerId,
            specialization,
            region,
            email,
            password,
            profile,
            // Add tenant reference from middleware
            tenant: req.tenantId
        });

        await fieldWorker.save();

        // Return field worker data (excluding password)
        const fieldWorkerData = {
            _id: fieldWorker._id,
            name: fieldWorker.name,
            workerId: fieldWorker.workerId,
            email: fieldWorker.email,
            specialization: fieldWorker.specialization,
            region: fieldWorker.region,
            tenant: fieldWorker.tenant,
            activeAssignments: fieldWorker.activeAssignments,
            profile: fieldWorker.profile || {},
            createdAt: fieldWorker.createdAt,
            status: status || 'Available',  // Status for UI (not stored in schema)
            personalEmail: fieldWorker.profile?.personalEmail || ''
        };

        res.status(201).json(fieldWorkerData);
    } catch (error) {
        res.status(500).json({ message: 'Error adding field worker', error: error.message });
    }
};

const getFieldWorkers = async (req, res) => {
    try {
        // Apply tenant filter - only get field workers for the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        let tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can see all field workers if no specific tenant is requested
            tenantFilter = {};
        }
        
        // Get all field workers
        const fieldWorkers = await FieldWorker.find(tenantFilter).select('-password');
        
        // Import the DamageReport model to count active assignments
        const DamageReport = require('../models/DamageReport');
        
        // Get active assignments count for each worker
        const workersWithAssignments = await Promise.all(fieldWorkers.map(async (worker) => {
            try {
                // Count active assignments (not Completed or Closed)
                const assignmentCount = await DamageReport.countDocuments({
                    assignedTo: worker._id,
                    status: { $nin: ['Completed', 'Closed'] }
                });
                
                // Create a new object with the worker data and assignment count
                const workerData = worker.toObject();
                workerData.activeAssignments = assignmentCount;
                
                // Determine status based on assignments and existing status
                if (assignmentCount >= 3) {
                    workerData.status = 'Busy';
                } else if (!workerData.status) {
                    workerData.status = 'Available';
                }
                
                return workerData;
            } catch (err) {
                console.error(`Error getting assignments for worker ${worker._id}:`, err);
                return worker.toObject();
            }
        }));
        
        res.status(200).json(workersWithAssignments);
    } catch (error) {
        console.error('Error fetching field workers:', error);
        res.status(500).json({ message: 'Error fetching field workers', error: error.message });
    }
};
const getFieldWorkerById = async (req, res) => {
    try {
        const { workerId } = req.params;
        
        // Apply tenant filter - only get field workers for the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can see any field worker
            tenantFilter = {};
        }
        
        const fieldWorker = await FieldWorker.findOne({ 
            workerId,
            ...tenantFilter
        }).select('-password');

        if (!fieldWorker) {
            return res.status(404).json({ message: 'Field worker not found' });
        }
        
        // Import DamageReport model
        const DamageReport = require('../models/DamageReport');
        
        // Count active assignments for this worker
        const assignmentCount = await DamageReport.countDocuments({
            assignedTo: fieldWorker._id,
            status: { $nin: ['Completed', 'Closed'] }
        });
        
        // Convert to plain object and add the assignment count
        const workerData = fieldWorker.toObject();
        workerData.activeAssignments = assignmentCount;
        
        // Determine status based on assignments and existing status
        if (assignmentCount >= 3) {
            workerData.status = 'Busy';
        } else if (!workerData.status) {
            workerData.status = 'Available';
        }

        res.status(200).json(workerData);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching field worker', error: error.message });
    }
};
// Update field worker details
// Get assignments for a specific field worker
const getFieldWorkerAssignments = async (req, res) => {
    try {
        const { workerId } = req.params;
        
        // Apply tenant filter
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            tenantFilter = {};
        }
        
        // Find the field worker first to validate existence
        const fieldWorker = await FieldWorker.findOne({
            workerId,
            ...tenantFilter
        });
        
        if (!fieldWorker) {
            return res.status(404).json({ message: 'Field worker not found' });
        }
        
        // Import DamageReport model
        const DamageReport = require('../models/DamageReport');
        
        // Get assignments for this worker
        const assignments = await DamageReport.find({
            assignedTo: fieldWorker._id,
            // Only include non-completed assignments by default, unless query param specifies otherwise
            ...(req.query.all !== 'true' ? { status: { $nin: ['Completed', 'Closed'] } } : {})
        }).sort({ createdAt: -1 }); // Sort by newest first
        
        res.status(200).json({
            fieldWorker: {
                id: fieldWorker._id,
                name: fieldWorker.name,
                workerId: fieldWorker.workerId,
                email: fieldWorker.email
            },
            assignments: assignments,
            count: assignments.length
        });
        
    } catch (error) {
        console.error('Error fetching field worker assignments:', error);
        res.status(500).json({ message: 'Error fetching assignments', error: error.message });
    }
};

const updateFieldWorker = async (req, res) => {
    try {
        const { workerId } = req.params;
        const updates = { ...req.body };
        
        // Handle profile fields properly - extract from updates and nest them
        const profileUpdates = {};
        
        if (updates.phone) {
            profileUpdates.phone = updates.phone;
            delete updates.phone;
        }
        
        if (updates.personalEmail !== undefined) {
            console.log('Updating personalEmail:', updates.personalEmail);
            profileUpdates.personalEmail = updates.personalEmail;
            profileUpdates.receiveDailyUpdates = updates.personalEmail ? true : false;
            delete updates.personalEmail;
        }
        
        if (updates.receiveDailyUpdates !== undefined) {
            console.log('Updating receiveDailyUpdates:', updates.receiveDailyUpdates);
            profileUpdates.receiveDailyUpdates = !!updates.receiveDailyUpdates;
            delete updates.receiveDailyUpdates;
        }
        
        // If we have any profile updates, merge them into the profile object
        if (Object.keys(profileUpdates).length > 0) {
            console.log('Profile updates to apply:', profileUpdates);
            // Make sure we have a $set operator, create if not exists
            updates.$set = updates.$set || {};
            
            // Add each profile field to the $set operator
            for (const [key, value] of Object.entries(profileUpdates)) {
                updates.$set[`profile.${key}`] = value;
            }
        }
        
        // Don't allow changing the tenant
        if (updates.tenant) {
            delete updates.tenant;
        }
        
        // Remove status as it's not in the schema (it's a UI field)
        if (updates.status) {
            delete updates.status;
        }
        
        // Apply tenant filter - only update field workers in the current tenant
        // req.tenantId is set by the ensureTenantIsolation middleware
        let tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Skip tenant filter for super-admin
        if (req.admin && req.admin.role === 'super-admin' && !req.query.tenant) {
            // Super admin can update any field worker
            tenantFilter = {};
        }

        const fieldWorker = await FieldWorker.findOneAndUpdate(
            { 
                workerId,
                ...tenantFilter 
            },
            updates,
            { new: true, runValidators: true }
        ).select('-password');

        if (!fieldWorker) {
            return res.status(404).json({ message: 'Field worker not found' });
        }

        // Add status for UI consistency if it was in the original request
        if (req.body.status) {
            fieldWorker._doc.status = req.body.status;
        }

        res.status(200).json(fieldWorker);
    } catch (error) {
        res.status(500).json({ message: 'Error updating field worker', error: error.message });
    }
};

// Function to send daily updates to field workers
const sendDailyUpdates = async (req, res) => {
    try {
        const { sendTestEmail } = req.query;
        const testWorkerId = req.params.workerId;
        
        // Apply tenant filter if applicable
        const tenantFilter = req.tenantId ? { tenant: req.tenantId } : {};
        
        // Find workers with personal emails who opted to receive daily updates
        const query = {
            'profile.personalEmail': { $exists: true, $ne: '' },
            'profile.receiveDailyUpdates': true,
            ...tenantFilter
        };
        
        // If this is a test email for a specific worker, modify the query
        if (sendTestEmail === 'true' && testWorkerId) {
            query.workerId = testWorkerId;
        }
        
        const fieldWorkers = await FieldWorker.find(query).select('-password');
        
        if (fieldWorkers.length === 0) {
            return res.status(404).json({ 
                message: sendTestEmail === 'true' ? 'Field worker not found or has no personal email configured' : 'No field workers with personal emails found'
            });
        }
        
        // Get the email service
        const { sendMail } = require('../utils/emailService');
        const DamageReport = require('../models/DamageReport');
        
        // Get current date for the report
        const today = new Date();
        const dateStr = today.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        // Send emails to each worker
        const emailResults = await Promise.all(fieldWorkers.map(async (worker) => {
            try {
                // Get worker's assigned reports
                const assignedReports = await DamageReport.find({
                    assignedTo: worker._id,
                    status: { $nin: ['Completed', 'Closed'] }
                }).select('reportId location damageType severity status createdAt');
                
                // Format the email content
                const emailSubject = `SafeStreets Daily Update - ${dateStr}`;
                
                let emailText = `Hello ${worker.name},\n\n`;
                emailText += `Here's your daily update for ${dateStr}:\n\n`;
                
                if (assignedReports.length === 0) {
                    emailText += "You currently have no active assignments.\n\n";
                } else {
                    emailText += `You have ${assignedReports.length} active assignments:\n\n`;
                    
                    assignedReports.forEach((report, index) => {
                        emailText += `${index + 1}. Report ID: ${report.reportId}\n`;
                        emailText += `   Location: ${report.location.address || 'Unknown'}\n`;
                        emailText += `   Damage Type: ${report.damageType}\n`;
                        emailText += `   Severity: ${report.severity}\n`;
                        emailText += `   Status: ${report.status}\n`;
                        emailText += `   Created: ${new Date(report.createdAt).toLocaleDateString()}\n\n`;
                    });
                }
                
                emailText += "To view more details or update the status of your assignments, please log in to the SafeStreets Field Worker application.\n\n";
                emailText += "Thank you for your hard work keeping our streets safe!\n\n";
                emailText += "Best regards,\nThe SafeStreets Team";
                
                // Send the email
                await sendMail({
                    to: worker.profile.personalEmail,
                    subject: emailSubject,
                    text: emailText
                });
                
                return {
                    workerId: worker.workerId,
                    name: worker.name,
                    email: worker.profile.personalEmail,
                    status: 'Success',
                    assignedReportsCount: assignedReports.length
                };
            } catch (emailError) {
                console.error(`Error sending email to ${worker.name}:`, emailError);
                return {
                    workerId: worker.workerId,
                    name: worker.name,
                    email: worker.profile.personalEmail,
                    status: 'Failed',
                    error: emailError.message
                };
            }
        }));
        
        // Count successes and failures
        const successCount = emailResults.filter(r => r.status === 'Success').length;
        const failureCount = emailResults.filter(r => r.status === 'Failed').length;
        
        res.status(200).json({
            message: `Daily updates processed: ${successCount} successful, ${failureCount} failed`,
            results: emailResults
        });
    } catch (error) {
        console.error('Error sending daily updates:', error);
        res.status(500).json({ 
            message: 'Error sending daily updates', 
            error: error.message 
        });
    }
};

module.exports = {
    addFieldWorker,
    getFieldWorkers,
    updateFieldWorker,
    getFieldWorkerById,
    getFieldWorkerAssignments,
    sendDailyUpdates
};