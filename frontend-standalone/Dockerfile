# Frontend Dockerfile for Enhanced Conspiracy Theory Debunker
FROM nginx:alpine

# Copy static files
COPY . /usr/share/nginx/html/

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create logs directory
RUN mkdir -p /var/log/nginx

# Set permissions
RUN chmod -R 755 /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]