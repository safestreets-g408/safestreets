const jwt = require('jsonwebtoken');
const FieldWorker = require('../models/FieldWorker');

const protectFieldWorker = async (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ message: 'No token, authorization denied' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Check if token contains fieldWorkerId
    if (!decoded.fieldWorkerId) {
      return res.status(401).json({ message: 'Invalid token format' });
    }
    
    // Verify field worker exists and is active
    const fieldWorker = await FieldWorker.findById(decoded.fieldWorkerId).select('-password');
    
    if (!fieldWorker) {
      return res.status(401).json({ message: 'Field worker not found' });
    }
    
    if (fieldWorker.profile && fieldWorker.profile.isActive === false) {
      return res.status(401).json({ message: 'Field worker account is deactivated' });
    }
    
    req.fieldWorker = { 
      id: decoded.fieldWorkerId,
      workerId: fieldWorker.workerId,
      name: fieldWorker.name,
      region: fieldWorker.region,
      specialization: fieldWorker.specialization
    };
    
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ message: 'Token has expired' });
    }
    res.status(401).json({ message: 'Token is not valid' });
  }
};

module.exports = { protectFieldWorker };
