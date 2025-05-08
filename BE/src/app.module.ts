import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { APP_GUARD } from '@nestjs/core';
import { JwtModule } from '@nestjs/jwt';
import access_tokenJwtConfig from './auth/@config/access_token-jwt.config';
import refresh_tokenJwtConfig from './auth/@config/refresh_token-jwt.config';
import { JwtAuthGuard } from './auth/@guard/jwt-auth.guard';
import { RolesGuard } from './auth/@guard/roles.guard';
import { AuthModule } from './auth/auth.module';
import { FileModule } from './file/file.module';
import { PrismaService } from './prisma.service';
import { UsersController } from './users/users.controller';
import { UsersModule } from './users/users.module';
import { ScheduleModule as NestScheduleModule } from '@nestjs/schedule';
import { AppointmentModule } from './appointment/appointment.module';
import { ScheduleModule } from './schedule/schedule.module';
import { PaymentModule } from './payment/payment.module';
import { PaymentController } from './payment/payment.controller';
import { PredictionModule } from './prediction/prediction.module';
import { StatisticsModule } from './statistics/statistics.module';

@Module({
  imports: [
    UsersModule,
    AuthModule,
    JwtModule.registerAsync(access_tokenJwtConfig.asProvider()),
    ConfigModule.forFeature(access_tokenJwtConfig),
    ConfigModule.forFeature(refresh_tokenJwtConfig),
    FileModule,
    NestScheduleModule.forRoot(),
    AppointmentModule,
    ScheduleModule,
    PaymentModule,
    PredictionModule,
    StatisticsModule,
  ],
  controllers: [UsersController, PaymentController],
  providers: [
    PrismaService,
    {
      provide: APP_GUARD,
      useClass: JwtAuthGuard,
    },
    {
      provide: APP_GUARD,
      useClass: RolesGuard,
    },
  ],
})
export class AppModule {}
