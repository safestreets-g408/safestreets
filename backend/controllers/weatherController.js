const fetch = require('node-fetch');

// Get weather information based on coordinates
const getWeatherInfo = async (req, res) => {
  try {
    const { lat, lon } = req.query;
    
    if (!lat || !lon) {
      return res.status(400).json({ message: 'Latitude and longitude are required' });
    }
    
    // You would normally use an API key from an environment variable
    // For this example, we're providing a mock response
    // In a real implementation, you'd call a weather API like:
    // const apiKey = process.env.WEATHER_API_KEY;
    // const weatherApiUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${apiKey}`;
    // const weatherResponse = await fetch(weatherApiUrl);
    // const weatherData = await weatherResponse.json();
    
    // Mock weather data
    const weatherData = generateMockWeather(lat, lon);
    
    res.status(200).json(weatherData);
  } catch (error) {
    console.error('Error fetching weather information:', error);
    res.status(500).json({ message: 'Error fetching weather information', error: error.message });
  }
};

// Generate mock weather data
const generateMockWeather = (lat, lon) => {
  // Generate semi-realistic weather based on current month
  const now = new Date();
  const month = now.getMonth(); // 0-11
  
  // Temperature range based on northern hemisphere seasons (adjust for southern hemisphere)
  let tempRange, conditionPool;
  
  if (month >= 2 && month <= 4) {
    // Spring
    tempRange = { min: 10, max: 25 };
    conditionPool = ['Clear', 'Clouds', 'Rain', 'Drizzle'];
  } else if (month >= 5 && month <= 7) {
    // Summer
    tempRange = { min: 18, max: 35 };
    conditionPool = ['Clear', 'Clouds', 'Thunderstorm', 'Drizzle'];
  } else if (month >= 8 && month <= 10) {
    // Fall
    tempRange = { min: 8, max: 22 };
    conditionPool = ['Clouds', 'Rain', 'Fog', 'Clear'];
  } else {
    // Winter
    tempRange = { min: -5, max: 15 };
    conditionPool = ['Snow', 'Clouds', 'Clear', 'Rain'];
  }
  
  // Generate temperature within range
  const temperature = Math.round(Math.random() * (tempRange.max - tempRange.min) + tempRange.min);
  
  // Select random condition
  const condition = conditionPool[Math.floor(Math.random() * conditionPool.length)];
  
  // Generate humidity and wind
  const humidity = Math.round(Math.random() * 50) + 30; // 30-80%
  const windSpeed = Math.round((Math.random() * 20 + 5) * 10) / 10; // 5-25 km/h with 1 decimal
  
  // Create response format similar to OpenWeatherMap
  return {
    coord: { lat: parseFloat(lat), lon: parseFloat(lon) },
    weather: [{
      id: getWeatherConditionId(condition),
      main: condition,
      description: getWeatherDescription(condition),
      icon: getWeatherIcon(condition)
    }],
    main: {
      temp: temperature,
      feels_like: Math.round(temperature - 2 + Math.random() * 4),
      temp_min: Math.round(temperature - 2 - Math.random() * 2),
      temp_max: Math.round(temperature + 2 + Math.random() * 2),
      pressure: Math.round(1000 + Math.random() * 30),
      humidity: humidity
    },
    wind: {
      speed: windSpeed,
      deg: Math.round(Math.random() * 360)
    },
    clouds: {
      all: condition === 'Clear' ? Math.round(Math.random() * 10) : Math.round(Math.random() * 50) + 50
    },
    visibility: condition === 'Fog' ? 1000 + Math.round(Math.random() * 4000) : 10000,
    dt: Math.floor(Date.now() / 1000),
    sys: {
      country: 'US',
      sunrise: Math.floor((now.setHours(6, 0, 0, 0)) / 1000),
      sunset: Math.floor((now.setHours(18, 0, 0, 0)) / 1000)
    },
    name: 'Current Location'
  };
};

// Helper function to get weather condition ID (similar to OpenWeatherMap)
const getWeatherConditionId = (condition) => {
  switch (condition) {
    case 'Clear': return 800;
    case 'Clouds': return 801 + Math.floor(Math.random() * 4);
    case 'Rain': return 500 + Math.floor(Math.random() * 4);
    case 'Thunderstorm': return 200 + Math.floor(Math.random() * 5);
    case 'Snow': return 600 + Math.floor(Math.random() * 4);
    case 'Drizzle': return 300 + Math.floor(Math.random() * 3);
    case 'Fog': return 741;
    default: return 800;
  }
};

// Helper function to get weather description
const getWeatherDescription = (condition) => {
  switch (condition) {
    case 'Clear': return 'clear sky';
    case 'Clouds': 
      const cloudTypes = ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'];
      return cloudTypes[Math.floor(Math.random() * cloudTypes.length)];
    case 'Rain':
      const rainTypes = ['light rain', 'moderate rain', 'heavy rain', 'very heavy rain'];
      return rainTypes[Math.floor(Math.random() * rainTypes.length)];
    case 'Thunderstorm': return 'thunderstorm';
    case 'Snow': return 'snow';
    case 'Drizzle': return 'light intensity drizzle';
    case 'Fog': return 'fog';
    default: return 'unknown';
  }
};

// Helper function to get weather icon code
const getWeatherIcon = (condition) => {
  // OpenWeatherMap-like icon codes
  switch (condition) {
    case 'Clear': return '01d';
    case 'Clouds': return ['02d', '03d', '04d'][Math.floor(Math.random() * 3)];
    case 'Rain': return ['09d', '10d'][Math.floor(Math.random() * 2)];
    case 'Thunderstorm': return '11d';
    case 'Snow': return '13d';
    case 'Drizzle': return '09d';
    case 'Fog': return '50d';
    default: return '01d';
  }
};

module.exports = {
  getWeatherInfo
};
