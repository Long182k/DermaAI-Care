<p align="center">
  <a href="http://nestjs.com/" target="blank"><img src="https://nestjs.com/img/logo-small.svg" width="120" alt="Nest Logo" /></a>
</p>

[circleci-image]: https://img.shields.io/circleci/build/github/nestjs/nest/master?token=abc123def456
[circleci-url]: https://circleci.com/gh/nestjs/nest

  <p align="center">A progressive <a href="http://nodejs.org" target="_blank">Node.js</a> framework for building efficient and scalable server-side applications.</p>
    <p align="center">
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/l/@nestjs/core.svg" alt="Package License" /></a>
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/dm/@nestjs/common.svg" alt="NPM Downloads" /></a>
<a href="https://circleci.com/gh/nestjs/nest" target="_blank"><img src="https://img.shields.io/circleci/build/github/nestjs/nest/master" alt="CircleCI" /></a>
<a href="https://coveralls.io/github/nestjs/nest?branch=master" target="_blank"><img src="https://coveralls.io/repos/github/nestjs/nest/badge.svg?branch=master#9" alt="Coverage" /></a>
<a href="https://discord.gg/G7Qnnhy" target="_blank"><img src="https://img.shields.io/badge/discord-online-brightgreen.svg" alt="Discord"/></a>
<a href="https://opencollective.com/nest#backer" target="_blank"><img src="https://opencollective.com/nest/backers/badge.svg" alt="Backers on Open Collective" /></a>
<a href="https://opencollective.com/nest#sponsor" target="_blank"><img src="https://opencollective.com/nest/sponsors/badge.svg" alt="Sponsors on Open Collective" /></a>
  <a href="https://paypal.me/kamilmysliwiec" target="_blank"><img src="https://img.shields.io/badge/Donate-PayPal-ff3f59.svg" alt="Donate us"/></a>
    <a href="https://opencollective.com/nest#sponsor"  target="_blank"><img src="https://img.shields.io/badge/Support%20us-Open%20Collective-41B883.svg" alt="Support us"></a>
  <a href="https://twitter.com/nestframework" target="_blank"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow" alt="Follow us on Twitter"></a>
</p>
  <!--[![Backers on Open Collective](https://opencollective.com/nest/backers/badge.svg)](https://opencollective.com/nest#backer)
  [![Sponsors on Open Collective](https://opencollective.com/nest/sponsors/badge.svg)](https://opencollective.com/nest#sponsor)-->

# BE-Social-Media

Backend service for Social Media platform built with NestJS, Prisma, and WebSocket, featuring intelligent content evaluation through Natural Language Processing.

## Key Features

### Content Evaluation with NLP

The platform implements advanced Natural Language Processing to:

- Analyze sentiment in user posts and comments
- Detect potential harmful or inappropriate content
- Provide content mood insights
- Help maintain a positive community environment

### Real-time Communication

- WebSocket integration for instant messaging
- Live notifications for user interactions
- Real-time content updates

### Media Management

- Cloudinary integration for media file handling
- Support for images and videos
- Secure file storage and delivery

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- PNPM package manager (You can use NPM OR Yarn if you want)
- PostgreSQL database

### Installation

```bash
# Clone the repository
git clone https://github.com/Long182k/BE-SOCIAL-MEDIA.git
cd BE-Social-Media

# Install dependencies
pnpm install

# Start the development server
pnpm start:dev
```

## Architecture

The application is built using NestJS framework with a modular architecture:

### Core Modules

1. **NLP Module**

   - Content sentiment analysis
   - Text classification
   - Mood detection
   - Content moderation

2. **Authentication Module**

   - JWT-based authentication
   - User session management
   - Security middleware

3. **Posts Module**

   - Content creation and management
   - Automatic sentiment analysis
   - Media attachment handling

4. **WebSocket Module**
   - Real-time messaging
   - Live notifications
   - User presence tracking

## Some API Endpoints Sample

### Content Management
- `POST /posts` - Create new post (with automatic sentiment analysis)
- `GET /posts` - Get all posts with pagination
- `GET /posts/:id` - Get specific post with sentiment data
- `PUT /posts/:id` - Update post
- `DELETE /posts/:id` - Delete post

### Group Management
- `POST /groups` - Create new group
- `GET /groups` - Get all groups with pagination
- `GET /groups/:id` - Get specific group details
- `PUT /groups/:id` - Update group information
- `DELETE /groups/:id` - Delete group
- `POST /groups/:id/join` - Join a group
- `POST /groups/:id/leave` - Leave a group
- `GET /groups/:id/members` - Get group members
- `POST /groups/:id/remove-member` - Remove member from group

### Group Posts
- `POST /group-posts/:groupId` - Create post in group
- `GET /group-posts/:groupId` - Get all posts in group
- `PUT /group-posts/:id` - Update group post
- `DELETE /group-posts/:id` - Delete group post
- `GET /group-posts/:groupId/feed` - Get group feed with pagination

### Events
- `POST /events` - Create new event
- `GET /events` - Get all events with pagination
- `GET /events/:id` - Get specific event details
- `PUT /events/:id` - Update event information
- `DELETE /events/:id` - Delete event
- `POST /events/:id/rsvp` - RSVP to an event
- `GET /events/:id/attendees` - Get event attendees
- `GET /events/upcoming` - Get upcoming events
- `GET /events/past` - Get past events


## Environment Variables

Create a `.env` file in the root directory:

```env
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
JWT_SECRET="your-jwt-secret"
CLOUDINARY_CLOUD_NAME="your-cloud-name"
CLOUDINARY_API_KEY="your-api-key"
CLOUDINARY_API_SECRET="your-api-secret"
NLP_API_KEY="your-nlp-service-key"
```

## Content Evaluation Flow

1. User creates/updates content
2. NLP service analyzes the content
3. Sentiment and appropriateness scores are generated
4. Content is stored with analysis results
5. Moderation flags are raised if needed

## Development

```bash
# development
$ pnpm run start

# watch mode
$ pnpm run start:dev

# production mode
$ pnpm run start:prod
```

## Testing

```bash
# unit tests
$ pnpm run test

# e2e tests
$ pnpm run test:e2e

# test coverage
$ pnpm run test:cov
```

## Resources

- [NestJS Documentation](https://docs.nestjs.com)
- [Natural Language Processing API Docs](https://your-nlp-service-docs)
- [WebSocket Integration Guide](https://docs.nestjs.com/websockets/gateways)

## Support

This project is open source and welcomes contributions. Feel free to submit issues and enhancement requests.

## License

This project is [MIT licensed](LICENSE).

## Stay in touch

- Author - [Kamil Myśliwiec](https://twitter.com/kammysliwiec)
- Website - [https://nestjs.com](https://nestjs.com/)
- Twitter - [@nestframework](https://twitter.com/nestframework)
