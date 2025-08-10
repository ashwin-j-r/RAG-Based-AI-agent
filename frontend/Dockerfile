# Stage 1: Build the React application
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies to leverage caching
COPY package.json ./
COPY package-lock.json ./
RUN npm install

# Copy the rest of the application source code
COPY . .

# Build the application for production
RUN npm run build

# Stage 2: Serve the built application with Nginx
FROM nginx:stable-alpine

# Copy the static files from the build stage
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx needs to be configured to support client-side routing
# We remove the default config and add our own
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d

# Expose port 80
EXPOSE 80

# The default Nginx command will start the server
CMD ["nginx", "-g", "daemon off;"]