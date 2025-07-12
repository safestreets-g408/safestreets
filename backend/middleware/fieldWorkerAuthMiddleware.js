const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const FieldWorker = require('../models/FieldWorker');
const Tenant = require('../models/Tenant');
const { getTokenFromCache } = require('../utils/jwtCache');

const protectFieldWorker = async (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ message: 'No token, authorization denied' });
  }

  try {
    const cachedToken = await getTokenFromCache(token);

    let decoded;
    if (cachedToken && cachedToken.userId) {
      decoded = { fieldWorkerId: cachedToken.userId };
    } else {
      decoded = jwt.verify(token, process.env.JWT_SECRET);

      if (!decoded.fieldWorkerId) {
        return res.status(401).json({ message: 'Invalid token format' });
      }
    }

    const fieldWorker = await FieldWorker.findById(decoded.fieldWorkerId)
      .select('-password')
      .populate('tenant');

    if (!fieldWorker) {
      return res.status(401).json({ message: 'Field worker not found' });
    }

    if (fieldWorker.profile && fieldWorker.profile.isActive === false) {
      return res.status(401).json({ message: 'Field worker account is deactivated' });
    }

    if (!fieldWorker.tenant) {
      return res.status(401).json({ message: 'No tenant associated with this account' });
    }

    if (fieldWorker.tenant._id && !fieldWorker.tenant.active) {
      return res.status(401).json({ message: 'Tenant account is inactive' });
    }

    if (typeof fieldWorker.tenant === 'string' || fieldWorker.tenant instanceof mongoose.Types.ObjectId) {
      const tenant = await Tenant.findById(fieldWorker.tenant);
      if (!tenant || !tenant.active) {
        return res.status(401).json({ message: 'Tenant account is inactive' });
      }
    }

    req.fieldWorker = {
      _id: decoded.fieldWorkerId,  // Use _id for consistency with MongoDB ObjectId
      id: decoded.fieldWorkerId,   // Keep id for backward compatibility
      workerId: fieldWorker.workerId,
      name: fieldWorker.name,
      region: fieldWorker.region,
      specialization: fieldWorker.specialization,
      tenant: fieldWorker.tenant._id || fieldWorker.tenant
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
