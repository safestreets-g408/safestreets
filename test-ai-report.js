const fetch = require('node-fetch');

// Test AI report upload with correct fields
async function testAiReportUpload() {
  const baseUrl = 'http://localhost:5030';
  
  // You would need a valid token from a field worker login
  // For testing, you can get one by logging in through the mobile app
  const token = 'YOUR_FIELD_WORKER_TOKEN_HERE';
  
  // Create a simple base64 test image (1x1 pixel)
  const testImageBase64 = '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A8A8A8A8A8A==';
  
  const requestBody = {
    damageType: 'POTHOLE',
    severity: 'HIGH',
    priority: 8,
    predictionClass: 'POTHOLE',
    annotatedImageBase64: testImageBase64,
    location: {
      coordinates: [-122.4194, 37.7749],
      address: '123 Test Street, San Francisco, CA'
    },
    description: 'Test pothole report from script'
  };
  
  console.log('Testing AI report upload...');
  console.log('Request body:', {
    ...requestBody,
    annotatedImageBase64: `[${testImageBase64.length} chars]`
  });
  
  try {
    const response = await fetch(`${baseUrl}/fieldworker/damage/ai-reports/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    console.log('Response status:', response.status);
    
    const data = await response.json();
    console.log('Response data:', data);
    
    if (response.ok) {
      console.log('✅ AI report upload successful!');
    } else {
      console.log('❌ AI report upload failed:', data.message);
      if (data.received) {
        console.log('Server received:', data.received);
      }
    }
  } catch (error) {
    console.error('❌ Request failed:', error.message);
  }
}

// Run the test
console.log('Note: You need to provide a valid field worker token to run this test.');
console.log('You can get one by logging in through the mobile app and checking the network requests.');
// testAiReportUpload();
