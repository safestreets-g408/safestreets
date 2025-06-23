# SafeStreets Backend

The SafeStreets backend API server provides endpoints for damage reporting, field worker management, and administrative functions.

## Prerequisites

- Node.js (v14+)
- MongoDB
- Redis (for caching)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
npm install
```

3. Create a `.env` file in the backend directory (use `.env.sample` as a template)

## Redis Setup

Redis is used for caching and improving application performance. To use Redis:

1. Install Redis on your system:
   - **macOS**: `brew install redis`
   - **Linux**: `sudo apt-get install redis-server`
   - **Windows**: Download from [Redis Windows](https://github.com/microsoftarchive/redis/releases)

2. Start the Redis server:
   - **macOS/Linux**: `redis-server`
   - **Windows**: Start the Redis service

3. Ensure your `.env` file has the Redis connection URL:
   ```
   REDIS_URL=redis://localhost:6379
   ```

## Caching Strategy

The application uses Redis for caching:

- JWT tokens for faster authentication
- API responses for frequently accessed data
- Weather information
- Damage report listings

## Running the Server

```bash
npm start
```

Or for development with auto-reload:

```bash
npm run dev
```

## Key Features

- Multi-tenant architecture
- JWT authentication with Redis caching
- Field worker management
- Damage report tracking
- Weather data integration

This project was created using `npm init`
