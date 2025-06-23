/**
 * Network connectivity test utility
 * This script tests connectivity to various local addresses to help troubleshoot
 * Expo Go connectivity issues
 * 
 * Run with: node scripts/networkTest.js
 */

const http = require('http');
const os = require('os');

// Get all local IP addresses
function getLocalIpAddresses() {
  const interfaces = os.networkInterfaces();
  const addresses = [];

  Object.keys(interfaces).forEach(interfaceName => {
    const networkInterface = interfaces[interfaceName];
    
    networkInterface.forEach(address => {
      if (!address.internal && address.family === 'IPv4') {
        addresses.push({
          interface: interfaceName,
          address: address.address,
        });
      }
    });
  });

  return addresses;
}

// Test connectivity to a specific host and port
function testConnectivity(host, port) {
  return new Promise((resolve) => {
    const req = http.request({
      host: host,
      port: port,
      path: '/',
      method: 'HEAD',
      timeout: 3000
    }, (res) => {
      console.log(`✅ Successfully connected to ${host}:${port} - Status: ${res.statusCode}`);
      resolve(true);
    });

    req.on('error', (err) => {
      console.log(`❌ Failed to connect to ${host}:${port} - Error: ${err.message}`);
      resolve(false);
    });

    req.on('timeout', () => {
      console.log(`❌ Connection to ${host}:${port} timed out`);
      req.destroy();
      resolve(false);
    });

    req.end();
  });
}

// Create a simple test server to check if others can connect to us
function startTestServer(port) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end('Network test server is running!');
    });
    
    server.on('error', (err) => {
      console.log(`❌ Could not start test server on port ${port}: ${err.message}`);
      resolve(null);
    });
    
    server.listen(port, () => {
      console.log(`✅ Test server running on port ${port}`);
      resolve(server);
    });
  });
}

async function runTests() {
  console.log('======= NETWORK CONNECTIVITY TESTS =======');
  console.log('Testing backend server connections...\n');
  
  // First test localhost connectivity
  await testConnectivity('localhost', 5030);
  await testConnectivity('127.0.0.1', 5030);
  
  // Test all local IP addresses
  const addresses = getLocalIpAddresses();
  console.log('\nTesting all local IP addresses:');
  
  for (const addr of addresses) {
    console.log(`\nTesting interface ${addr.interface} (${addr.address}):`);
    await testConnectivity(addr.address, 5030);
  }

  // Start a test server to check inbound connectivity
  console.log('\n\nStarting test server on port 8081 (same as Expo)...');
  const server = await startTestServer(8081);
  
  if (server) {
    console.log('\nTest server is now running.');
    console.log('On your device running Expo Go, try to open:');
    addresses.forEach(addr => {
      console.log(`http://${addr.address}:8081`);
    });
    console.log('\nIf any of these URLs work in your device browser, use that IP address in your config.js');
    console.log('Press Ctrl+C to stop the server when done testing');
  }
}

// Run the tests
runTests();
