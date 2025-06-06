import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';
import { User } from '@prisma/client';
import { CurrentUser } from '../auth/@decorator/current-user.decorator';
import { JwtAuthGuard } from '../auth/@guard/jwt-auth.guard';
import { RolesGuard } from '../auth/@guard/roles.guard';
import { AppointmentService } from './appointment.service';
import { CreateAppointmentDto } from './dto/create-appointment.dto';
import { UpdateAppointmentDto } from './dto/update-appointment.dto';
import { ChangeAppointmentStatusDto } from './dto/change-appointment-status.dto';

@Controller('appointments')
@UseGuards(JwtAuthGuard, RolesGuard)
export class AppointmentController {
  constructor(private readonly appointmentService: AppointmentService) {}

  @Post()
  async createAppointment(
    @Body() createAppointmentDto: CreateAppointmentDto,
    @CurrentUser('userId') userId: string,
  ) {
    return await this.appointmentService.createAppointment(
      createAppointmentDto,
      userId,
    );
  }

  @Get('user')
  async getUserAppointments(@CurrentUser() currentUser: User) {
    return await this.appointmentService.findUserAppointments(
      currentUser.id ?? currentUser.userId,
      currentUser.role,
    );
  }

  @Get('history')
  async getAppointmentHistory(@CurrentUser() currentUser: User) {
    return await this.appointmentService.findAppointmentHistory(
      currentUser.id ?? currentUser.userId,
      currentUser.role,
    );
  }

  @Get(':id')
  async getAppointmentById(@Param('id') id: string) {
    return await this.appointmentService.findAppointmentById(id);
  }

  @Patch(':id')
  async updateAppointment(
    @Param('id') id: string,
    @Body() updateAppointmentDto: UpdateAppointmentDto,
    @CurrentUser() currentUser: User,
  ) {
    return await this.appointmentService.updateAppointment(
      id,
      updateAppointmentDto,
      currentUser,
    );
  }

  @Patch(':id/cancel')
  async cancelAppointment(
    @Param('id') id: string,
    @CurrentUser() currentUser: User,
  ) {
    return await this.appointmentService.cancelAppointment(id, currentUser);
  }

  @Patch(':id/status')
  async changeAppointmentStatus(
    @Param('id') id: string,
    @Body() changeStatusDto: ChangeAppointmentStatusDto,
    @CurrentUser() currentUser: User,
  ) {
    return await this.appointmentService.changeAppointmentStatus(
      id,
      changeStatusDto.status,
      currentUser,
    );
  }
}
