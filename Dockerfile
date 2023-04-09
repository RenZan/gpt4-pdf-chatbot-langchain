FROM node:18.15.0
RUN npm install -g npm@latest
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
