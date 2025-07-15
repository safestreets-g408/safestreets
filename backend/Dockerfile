FROM node:20-slim

# Set working directory
WORKDIR /usr/src/app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Create upload and results directories
RUN mkdir -p uploads results

# Expose port (default: 3001)
EXPOSE 5030

# Start the server
CMD ["node", "server.js"] 