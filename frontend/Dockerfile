# Multi-stage build: compile the Vite app in a Node image, then serve the
# resulting static files from a tiny nginx image. Why two stages: the
# final image doesn't need Node, npm, or any of the ~200MB of
# node_modules — just the built HTML/JS/CSS bundle, which nginx serves
# far more efficiently than `npm run dev` ever would in production.

FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
