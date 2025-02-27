import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { UsersModule } from 'src/users/users.module';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtStrategy } from 'src/auth/@strategies/jwt.strategy';
import { LocalStrategy } from './@strategies/local.strategy';
import { ConfigModule } from '@nestjs/config';
import jwtConfig from './@config/access_token-jwt.config';
import refreshJwtConfig from './@config/refresh_token-jwt.config';
import { MailerModule } from '@nestjs-modules/mailer';
import { HandlebarsAdapter } from '@nestjs-modules/mailer/dist/adapters/handlebars.adapter';

import 'dotenv/config';
import { RefreshJwtStrategy } from './@strategies/refresh-jwt.strategy';
import { UserRepository } from 'src/users/users.repository';
import { PrismaService } from 'src/prisma.service';

@Module({
  imports: [
    UsersModule,
    PassportModule,
    JwtModule.registerAsync(jwtConfig.asProvider()),
    ConfigModule.forFeature(jwtConfig),
    ConfigModule.forFeature(refreshJwtConfig),
    MailerModule.forRoot({
      transport: {
        host: process.env.EMAIL_HOST,
        port: Number(process.env.EMAIL_PORT),
        secure: true,
        auth: {
          user: process.env.EMAIL_USER,
          pass: process.env.EMAIL_PASSWORD,
        },
      },
      defaults: {
        from: `"Friendzii Social Media" <${process.env.EMAIL_USER}>`,
      },
      template: {
        dir: process.cwd() + '/src/mail/templates',
        adapter: new HandlebarsAdapter(),
        options: {
          strict: true,
        },
      },
    }),
  ],
  controllers: [AuthController],
  providers: [
    AuthService,
    LocalStrategy,
    JwtStrategy,
    RefreshJwtStrategy,
    UserRepository,
    PrismaService,
  ],
  exports: [AuthService],
})
export class AuthModule {}
