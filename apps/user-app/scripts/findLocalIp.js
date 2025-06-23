/**
 * Helper script to print your local network IP addresses
 * Run with: node scripts/findLocalIp.js
 */

const os = require('os');

function getLocalIpAddresses() {
  const interfaces = os.networkInterfaces();
  const addresses = [];

  Object.keys(interfaces).forEach(interfaceName => {
    const networkInterface = interfaces[interfaceName];
    
    networkInterface.forEach(address => {
      // Skip internal/loopback addresses and IPv6 addresses
      if (!address.internal && address.family === 'IPv4') {
        addresses.push({
          interface: interfaceName,
          address: address.address,
          cidr: address.cidr
        });
      }
    });
  });

  return addresses;
}

console.log('Your local network IP addresses:');
console.log('---------------------------------');

const addresses = getLocalIpAddresses();
if (addresses.length === 0) {
  console.log('No external IPv4 addresses found.');
} else {
  addresses.forEach(addr => {
    console.log(`Interface: ${addr.interface}`);
    console.log(`IP Address: ${addr.address}`);
    console.log(`CIDR: ${addr.cidr}`);
    console.log(`For API_BASE_URL: http://${addr.address}:5030/api`);
    console.log('---------------------------------');
  });
  
  console.log('\nInstructions:');
  console.log('1. Copy one of the API_BASE_URL values above');
  console.log('2. In config.js, update the API_BASE_URL with this value');
  console.log('3. Make sure your server is running and accessible on port 5030');
}
